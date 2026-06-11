"""Per-device compute queue for training and evaluation jobs."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, Dict, List, Literal, Optional, Tuple

from pe_common.devices import AUTO_DEVICE, list_accelerator_ids, list_device_ids, resolve_device_id

from .job_cancel import (
    JobCancelledError,
    clear_cancel,
    is_cancel_requested,
    register_cancel_event,
    request_cancel,
)
from ..evaluation.jobs import get_job as get_eval_job
from ..evaluation.jobs import mark_cancelled as mark_eval_cancelled
from ..evaluation.jobs import mark_failed as mark_eval_failed
from ..evaluation.jobs import mark_stopping as mark_eval_stopping
from ..evaluation.jobs import update_job as update_eval_job
from ..evaluation.runner import execute_evaluation
from ..evaluation.schemas import EvaluationRequest
from ..training.jobs import get_job as get_train_job
from ..training.jobs import mark_cancelled as mark_train_cancelled
from ..training.jobs import mark_failed as mark_train_failed
from ..training.jobs import mark_stopping as mark_train_stopping
from ..training.jobs import update_job as update_train_job
from ..training.runner import execute_training
from ..training.schemas import TrainingRequest

logger = logging.getLogger(__name__)

JobKind = Literal["train", "evaluate"]
QueuedJob = Tuple[JobKind, str]


class ComputeDeviceScheduler:
    """Assign queued jobs to devices; at most one active job per device."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._wait_queue: Deque[QueuedJob] = deque()
        self._running_on_device: Dict[str, Optional[QueuedJob]] = {}
        self._job_device: Dict[QueuedJob, str] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(list_device_ids(include_cpu=True))),
            thread_name_prefix="pe-compute",
        )
        self._refresh_device_map()

    def _refresh_device_map(self) -> None:
        """Ensure each discovered device has a slot in the running map."""
        for device_id in list_device_ids(include_cpu=True):
            self._running_on_device.setdefault(device_id, None)

    def submit_training(self, job_id: str, request: TrainingRequest) -> None:
        self._submit(job_id, "train", request.device or AUTO_DEVICE)

    def submit_evaluation(self, job_id: str, request: EvaluationRequest) -> None:
        self._submit(job_id, "evaluate", request.device or AUTO_DEVICE)

    def _submit(self, job_id: str, kind: JobKind, requested: str) -> None:
        queued: QueuedJob = (kind, job_id)
        with self._lock:
            self._refresh_device_map()
            self._update_job_locked(kind, job_id, device_requested=requested, device_assigned=None, queue_position=None)
            if requested in (None, AUTO_DEVICE) and not list_accelerator_ids():
                error = "No accelerator devices available; auto assignment does not use CPU"
                if kind == "train":
                    mark_train_failed(job_id, error)
                else:
                    mark_eval_failed(job_id, error)
                logger.error("Job %s/%s rejected: %s", kind, job_id, error)
                return
            device_id = self._try_assign_locked(queued, requested)
            if device_id is None:
                self._wait_queue.append(queued)
                self._update_queue_positions_locked()
                logger.info("Job %s/%s waiting for device (requested=%s)", kind, job_id, requested)
                return
            self._launch_locked(queued, device_id)

    def cancel_job(self, kind: JobKind, job_id: str) -> bool:
        """Stop a queued or running job. Returns True if the job was known to the scheduler."""
        queued: QueuedJob = (kind, job_id)
        with self._lock:
            new_queue: Deque[QueuedJob] = deque()
            removed_from_queue = False
            for item in self._wait_queue:
                if item == queued:
                    removed_from_queue = True
                    continue
                new_queue.append(item)
            if removed_from_queue:
                self._wait_queue = new_queue
                if kind == "train":
                    mark_train_cancelled(job_id)
                else:
                    mark_eval_cancelled(job_id)
                self._update_queue_positions_locked()
                self._dispatch_locked()
                return True

            if queued in self._job_device:
                request_cancel(kind, job_id)
                self._mark_stopping_if_active(kind, job_id)
                return True

            for running in self._running_on_device.values():
                if running == queued:
                    request_cancel(kind, job_id)
                    self._mark_stopping_if_active(kind, job_id)
                    return True
        return False

    def _mark_stopping_if_active(self, kind: JobKind, job_id: str) -> None:
        try:
            manifest = self._get_job_manifest(kind, job_id)
        except FileNotFoundError:
            return
        if manifest.get("status") not in ("queued", "running", "stopping"):
            return
        if kind == "train":
            mark_train_stopping(job_id)
        else:
            mark_eval_stopping(job_id)

    def device_snapshot(self) -> List[Dict[str, object]]:
        with self._lock:
            self._refresh_device_map()
            return [
                {
                    "device_id": device_id,
                    "running_job_id": self._format_job_id(self._running_on_device.get(device_id)),
                    "running_job_kind": self._running_on_device.get(device_id, (None, None))[0]
                    if self._running_on_device.get(device_id)
                    else None,
                    "queued_jobs": sum(
                        1
                        for kind, queued_id in self._wait_queue
                        if self._job_waits_for_device(device_id, kind, queued_id)
                    ),
                }
                for device_id in list_device_ids(include_cpu=True)
            ]

    def _job_waits_for_device(self, device_id: str, kind: JobKind, queued_id: str) -> bool:
        requested = self._get_job_manifest(kind, queued_id).get("device_requested")
        if requested in (None, AUTO_DEVICE):
            return device_id in list_device_ids(include_cpu=False)
        return requested == device_id

    def _format_job_id(self, queued: Optional[QueuedJob]) -> Optional[str]:
        if queued is None:
            return None
        return queued[1]

    def _get_job_manifest(self, kind: JobKind, job_id: str) -> Dict[str, object]:
        if kind == "train":
            return get_train_job(job_id)
        return get_eval_job(job_id)

    def _update_job_locked(self, kind: JobKind, job_id: str, **fields: object) -> None:
        if kind == "train":
            update_train_job(job_id, **fields)
        else:
            update_eval_job(job_id, **fields)

    def _try_assign_locked(self, queued: QueuedJob, requested: str) -> Optional[str]:
        if requested not in (None, AUTO_DEVICE):
            resolved = resolve_device_id(requested)
            if self._running_on_device.get(resolved) is None:
                self._running_on_device[resolved] = queued
                self._job_device[queued] = resolved
                return resolved
            return None

        for device_id in list_device_ids(include_cpu=False):
            if self._running_on_device.get(device_id) is None:
                self._running_on_device[device_id] = queued
                self._job_device[queued] = device_id
                return device_id
        return None

    def _launch_locked(self, queued: QueuedJob, device_id: str) -> None:
        kind, job_id = queued
        register_cancel_event(kind, job_id)
        self._update_job_locked(kind, job_id, device_assigned=device_id, queue_position=None)
        self._executor.submit(self._run_job, queued, device_id)

    def _run_job(self, queued: QueuedJob, device_id: str) -> None:
        kind, job_id = queued
        try:
            if is_cancel_requested(kind, job_id):
                raise JobCancelledError(f"Job {job_id} cancelled before start")
            request = _load_request(kind, job_id)
            if kind == "train":
                execute_training(request, job_id=job_id, device_id=device_id)
            else:
                execute_evaluation(request, job_id=job_id, device_id=device_id)
        except JobCancelledError:
            logger.info("%s job %s cancelled on %s", kind, job_id, device_id)
            try:
                if kind == "train":
                    mark_train_cancelled(job_id)
                else:
                    mark_eval_cancelled(job_id)
            except FileNotFoundError:
                pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s job %s failed on %s", kind, job_id, device_id)
            self._mark_failed_if_still_running(kind, job_id, exc)
        finally:
            clear_cancel(kind, job_id)
            with self._lock:
                self._release_locked(queued, device_id)
                self._dispatch_locked()

    def _mark_failed_if_still_running(self, kind: JobKind, job_id: str, exc: BaseException) -> None:
        """Ensure the manifest leaves running if the worker raised unexpectedly."""
        try:
            manifest = self._get_job_manifest(kind, job_id)
        except FileNotFoundError:
            return
        if manifest.get("status") != "running":
            return
        error = str(exc)
        if kind == "train":
            mark_train_failed(job_id, error)
        else:
            mark_eval_failed(job_id, error)

    def _release_locked(self, queued: QueuedJob, device_id: str) -> None:
        if self._running_on_device.get(device_id) == queued:
            self._running_on_device[device_id] = None
        self._job_device.pop(queued, None)

    def _dispatch_locked(self) -> None:
        if not self._wait_queue:
            self._update_queue_positions_locked()
            return

        pending: Deque[QueuedJob] = deque()
        while self._wait_queue:
            queued = self._wait_queue.popleft()
            kind, job_id = queued
            manifest = self._get_job_manifest(kind, job_id)
            if manifest.get("status") not in ("queued",):
                continue
            device_id = self._try_assign_locked(queued, manifest.get("device_requested", AUTO_DEVICE))
            if device_id is None:
                pending.append(queued)
                continue
            self._launch_locked(queued, device_id)

        self._wait_queue = pending
        self._update_queue_positions_locked()

    def _update_queue_positions_locked(self) -> None:
        for index, (kind, job_id) in enumerate(self._wait_queue, start=1):
            self._update_job_locked(kind, job_id, queue_position=index, device_assigned=None)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop accepting work and release worker threads."""
        with self._lock:
            while self._wait_queue:
                kind, job_id = self._wait_queue.popleft()
                try:
                    if kind == "train":
                        mark_train_cancelled(job_id)
                    else:
                        mark_eval_cancelled(job_id)
                except FileNotFoundError:
                    pass
            for kind, job_id in list(self._job_device.keys()):
                request_cancel(kind, job_id)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


def _load_request(kind: JobKind, job_id: str) -> TrainingRequest | EvaluationRequest:
    if kind == "train":
        from ..training.config import jobs_root

        request_path = jobs_root() / job_id / "request.json"
    else:
        from ..evaluation.config import eval_jobs_root

        request_path = eval_jobs_root() / job_id / "request.json"
    with open(request_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if kind == "train":
        return TrainingRequest.model_validate(payload)
    return EvaluationRequest.model_validate(payload)


_scheduler: Optional[ComputeDeviceScheduler] = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> ComputeDeviceScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = ComputeDeviceScheduler()
        return _scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None

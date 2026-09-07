"""Per-device compute queue for training, evaluation, and ensemble jobs."""
from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Literal, Optional, Tuple

from pe_common.devices import AUTO_DEVICE, list_accelerator_ids, list_device_ids, resolve_device_id

from .job_cancel import (
    JobCancelledError,
    clear_cancel,
    is_cancel_requested,
    register_cancel_event,
    request_cancel,
)
from .job_store import JobStore
from ..ensemble.jobs import store as ensemble_store
from ..ensemble.runner import execute_ensemble
from ..ensemble.schemas import EnsembleRequest
from ..evaluation.jobs import store as eval_store
from ..evaluation.runner import execute_evaluation
from ..evaluation.schemas import EvaluationRequest
from ..training.jobs import store as train_store
from ..training.runner import execute_training
from ..training.schemas import TrainingRequest
from ..training.tune_jobs import store as tune_store
from ..training.tune_study import execute_tuning
from ..training.tuning_schemas import TuningRequest

logger = logging.getLogger(__name__)

JobKind = Literal["train", "evaluate", "ensemble", "tune"]
QueuedJob = Tuple[JobKind, str]


@dataclass(frozen=True)
class _KindOps:
    store: JobStore
    request_type: type
    execute: Callable[..., Any]


def _kind_ops(kind: JobKind) -> _KindOps:
    # Built at call time so tests can monkeypatch execute_* on this module.
    return {
        "train": _KindOps(train_store, TrainingRequest, execute_training),
        "evaluate": _KindOps(eval_store, EvaluationRequest, execute_evaluation),
        "ensemble": _KindOps(ensemble_store, EnsembleRequest, execute_ensemble),
        "tune": _KindOps(tune_store, TuningRequest, execute_tuning),
    }[kind]


def _get_job_manifest(kind: JobKind, job_id: str) -> Dict[str, object]:
    return _kind_ops(kind).store.get(job_id)


def _update_job(kind: JobKind, job_id: str, **fields: object) -> None:
    _kind_ops(kind).store.update(job_id, **fields)


def _mark_cancelled(kind: JobKind, job_id: str) -> None:
    _kind_ops(kind).store.mark_cancelled(job_id)


def _mark_failed(kind: JobKind, job_id: str, error: str) -> None:
    _kind_ops(kind).store.mark_failed(job_id, error)


def _mark_stopping(kind: JobKind, job_id: str) -> None:
    _kind_ops(kind).store.mark_stopping(job_id)


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

    def submit_ensemble(self, job_id: str, request: EnsembleRequest) -> None:
        self._submit(job_id, "ensemble", request.device or AUTO_DEVICE)

    def submit_tuning(self, job_id: str, request: TuningRequest) -> None:
        self._submit(job_id, "tune", request.training.device or AUTO_DEVICE)

    def _submit(self, job_id: str, kind: JobKind, requested: str) -> None:
        queued: QueuedJob = (kind, job_id)
        with self._lock:
            self._refresh_device_map()
            _update_job(kind, job_id, device_requested=requested, device_assigned=None, queue_position=None)
            if requested in (None, AUTO_DEVICE) and not list_accelerator_ids():
                error = "No accelerator devices available; auto assignment does not use CPU"
                _mark_failed(kind, job_id, error)
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
                _mark_cancelled(kind, job_id)
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
            manifest = _get_job_manifest(kind, job_id)
        except FileNotFoundError:
            return
        if manifest.get("status") not in ("queued", "running", "stopping"):
            return
        _mark_stopping(kind, job_id)

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
        requested = _get_job_manifest(kind, queued_id).get("device_requested")
        if requested in (None, AUTO_DEVICE):
            return device_id in list_device_ids(include_cpu=False)
        return requested == device_id

    def _format_job_id(self, queued: Optional[QueuedJob]) -> Optional[str]:
        if queued is None:
            return None
        return queued[1]

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
        _update_job(kind, job_id, device_assigned=device_id, queue_position=None)
        self._executor.submit(self._run_job, queued, device_id)

    def _run_job(self, queued: QueuedJob, device_id: str) -> None:
        kind, job_id = queued
        try:
            if is_cancel_requested(kind, job_id):
                raise JobCancelledError(f"Job {job_id} cancelled before start")
            ops = _kind_ops(kind)
            request = ops.request_type.model_validate(ops.store.load_request_json(job_id))
            ops.execute(request, job_id=job_id, device_id=device_id)
        except JobCancelledError:
            logger.info("%s job %s cancelled on %s", kind, job_id, device_id)
            try:
                _mark_cancelled(kind, job_id)
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
            manifest = _get_job_manifest(kind, job_id)
        except FileNotFoundError:
            return
        if manifest.get("status") != "running":
            return
        _mark_failed(kind, job_id, str(exc))

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
            manifest = _get_job_manifest(kind, job_id)
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
            _update_job(kind, job_id, queue_position=index, device_assigned=None)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop accepting work and release worker threads."""
        with self._lock:
            while self._wait_queue:
                kind, job_id = self._wait_queue.popleft()
                try:
                    _mark_cancelled(kind, job_id)
                except FileNotFoundError:
                    pass
            for kind, job_id in list(self._job_device.keys()):
                request_cancel(kind, job_id)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


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

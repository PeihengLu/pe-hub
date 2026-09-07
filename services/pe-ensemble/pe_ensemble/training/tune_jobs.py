"""Filesystem-backed hyperparameter tuning job registry."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ..compute.job_store import JobStore
from .config import tune_jobs_root
from .tuning_schemas import TuningJobSummary, TuningRequest

LOG_FILENAME = "tune.log"
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

store = JobStore(
    root=tune_jobs_root,
    log_filename=LOG_FILENAME,
    unknown_message="Unknown tuning job: {job_id}",
    terminal_statuses=TERMINAL_JOB_STATUSES,
)


def create_job(request: TuningRequest, *, job_id: Optional[str] = None) -> str:
    training = request.training
    return store.create(
        {
            "job_kind": "tune",
            "status": "queued",
            "model_name": training.model_name.strip().lower(),
            "dataset_name": training.dataset_name,
            "n_trials": request.n_trials,
            "study_name": request.study_name,
            "started_at": None,
            "finished_at": None,
            "device_requested": training.device or "auto",
            "device_assigned": None,
            "queue_position": None,
            "best_trial": None,
            "best_value": None,
            "preset_path": None,
            "error": None,
            "result": None,
        },
        job_id=job_id,
        request_payload=request.model_dump(),
        exists_message="Tuning job already exists: {job_id}",
    )


def get_job(job_id: str) -> Dict[str, Any]:
    return store.get(job_id)


def list_jobs(*, limit: int = 50) -> List[Dict[str, Any]]:
    return store.list(limit=limit)


def update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    return store.update(job_id, **fields)


def mark_running(job_id: str) -> Dict[str, Any]:
    return store.mark_running(job_id, kind_label="Tuning")


def mark_stopping(job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
    return store.mark_stopping(job_id, reason=reason)


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return store.mark_terminal(
        job_id,
        "succeeded",
        study_name=result.get("study_name"),
        best_trial=result.get("best_trial"),
        best_value=result.get("best_value"),
        preset_path=result.get("preset_path"),
        result=result,
        error=None,
    )


def mark_failed(job_id: str, error: str) -> Dict[str, Any]:
    return store.mark_failed(job_id, error)


def mark_cancelled(job_id: str, *, reason: str = "Cancelled by user") -> Dict[str, Any]:
    return store.mark_cancelled(job_id, reason=reason)


def delete_job(job_id: str) -> None:
    store.delete(job_id)


def append_log(job_id: str, message: str) -> None:
    store.append_log(job_id, message)


def read_logs(job_id: str, *, offset: int = 0) -> tuple[str, int]:
    return store.read_logs(job_id, offset=offset)


def wait_for_job(
    job_id: str,
    *,
    poll_interval: float = 0.5,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    return store.wait_for_job(
        job_id,
        poll_interval=poll_interval,
        timeout=timeout,
        timeout_message=f"Tuning job {job_id} did not finish within {timeout}s",
    )


def job_summary(manifest: Dict[str, Any]) -> TuningJobSummary:
    return TuningJobSummary(
        job_id=manifest["job_id"],
        status=manifest["status"],
        model_name=manifest["model_name"],
        dataset_name=manifest["dataset_name"],
        n_trials=manifest["n_trials"],
        study_name=manifest.get("study_name"),
        created_at=manifest["created_at"],
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        device_requested=manifest.get("device_requested"),
        device_assigned=manifest.get("device_assigned"),
        queue_position=manifest.get("queue_position"),
        best_trial=manifest.get("best_trial"),
        best_value=manifest.get("best_value"),
        preset_path=manifest.get("preset_path"),
        error=manifest.get("error"),
    )


@contextmanager
def job_log_context(job_id: str):
    with store.job_log_context(job_id):
        yield

"""Filesystem-backed evaluation job registry."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ..compute.job_store import JobStore
from .config import eval_jobs_root
from .schemas import EvaluationJobSummary, EvaluationRequest

LOG_FILENAME = "eval.log"
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled", "skipped"})

store = JobStore(
    root=eval_jobs_root,
    log_filename=LOG_FILENAME,
    unknown_message="Unknown evaluation job: {job_id}",
    terminal_statuses=TERMINAL_JOB_STATUSES,
)


def create_job(request: EvaluationRequest, *, job_id: Optional[str] = None) -> str:
    return store.create(
        {
            "job_kind": "evaluate",
            "status": "queued",
            "model_name": request.model_name.strip().lower(),
            "benchmark_name": request.benchmark_name,
            "started_at": None,
            "finished_at": None,
            "device_requested": request.device or "auto",
            "device_assigned": None,
            "queue_position": None,
            "weights_id": request.weights,
            "error": None,
            "result": None,
        },
        job_id=job_id,
        request_payload=request.model_dump(),
        exists_message="Evaluation job already exists: {job_id}",
    )


def get_job(job_id: str) -> Dict[str, Any]:
    return store.get(job_id)


def list_jobs(*, limit: int = 50) -> List[Dict[str, Any]]:
    return store.list(limit=limit)


def update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    return store.update(job_id, **fields)


def mark_running(job_id: str) -> Dict[str, Any]:
    return store.mark_running(job_id, kind_label="Evaluation")


def mark_stopping(job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
    return store.mark_stopping(job_id, reason=reason)


def _weights_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    resolved = result.get("weights") or result.get("weights_id")
    return {"weights_id": resolved} if resolved else {}


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return store.mark_terminal(
        job_id,
        "succeeded",
        result=result,
        error=None,
        **_weights_fields(result),
    )


def mark_skipped(job_id: str, result: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    return store.mark_terminal(
        job_id,
        "skipped",
        result=result,
        error=reason,
        **_weights_fields(result),
    )


def mark_failed(
    job_id: str,
    error: str,
    *,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if result is not None:
        fields["result"] = result
    return store.mark_failed(job_id, error, **fields)


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
    return store.wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)


def job_summary(manifest: Dict[str, Any]) -> EvaluationJobSummary:
    return EvaluationJobSummary(
        job_id=manifest["job_id"],
        status=manifest["status"],
        model_name=manifest["model_name"],
        benchmark_name=manifest["benchmark_name"],
        created_at=manifest["created_at"],
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        device_requested=manifest.get("device_requested"),
        device_assigned=manifest.get("device_assigned"),
        queue_position=manifest.get("queue_position"),
        weights_id=manifest.get("weights_id"),
        error=manifest.get("error"),
    )


@contextmanager
def job_log_context(job_id: str):
    with store.job_log_context(job_id):
        yield

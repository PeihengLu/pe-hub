"""Filesystem-backed plugin validation job registry."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from ..compute.job_store import JobStore
from .config import validation_jobs_root

LOG_FILENAME = "validation.log"
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled", "skipped"})

store = JobStore(
    root=validation_jobs_root,
    log_filename=LOG_FILENAME,
    unknown_message="Unknown plugin validation job: {job_id}",
    request_filename=None,
    terminal_statuses=TERMINAL_JOB_STATUSES,
)


def create_job(plugin_name: str, *, job_id: Optional[str] = None) -> str:
    return store.create(
        {
            "job_kind": "plugin_validate",
            "status": "queued",
            "plugin_name": plugin_name.strip().lower(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        },
        job_id=job_id,
        exists_message="Validation job already exists: {job_id}",
    )


def get_job(job_id: str) -> Dict[str, Any]:
    return store.get(job_id)


def list_jobs_for_plugin(plugin_name: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    key = plugin_name.strip().lower()
    return store.list(limit=limit, match=lambda item: item.get("plugin_name") == key)


def find_active_job(plugin_name: str) -> Optional[Dict[str, Any]]:
    for manifest in list_jobs_for_plugin(plugin_name, limit=20):
        if manifest.get("status") in ("queued", "running", "stopping"):
            return manifest
    return None


def update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    return store.update(job_id, **fields)


def mark_running(job_id: str) -> Dict[str, Any]:
    return store.mark_running(job_id, kind_label="Validation")


def mark_stopping(job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
    return store.mark_stopping(job_id, reason=reason)


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return store.mark_terminal(job_id, "succeeded", result=result, error=None)


def mark_failed(job_id: str, error: str) -> Dict[str, Any]:
    return store.mark_failed(job_id, error, result=None)


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
        timeout_message=f"Validation job {job_id} did not finish within {timeout}s",
    )


@contextmanager
def job_log_context(job_id: str):
    with store.job_log_context(job_id):
        yield

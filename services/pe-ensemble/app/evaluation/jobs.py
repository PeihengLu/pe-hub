"""Filesystem-backed evaluation job registry."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..compute.job_cancel import JobCancelledError
from .config import eval_jobs_root
from .schemas import EvaluationJobSummary, EvaluationRequest

MANIFEST_FILENAME = "manifest.json"
LOG_FILENAME = "eval.log"
REQUEST_FILENAME = "request.json"
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled", "skipped"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _job_dir(job_id: str) -> Path:
    return eval_jobs_root() / job_id


def _write_manifest(job_dir: Path, manifest: Dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_job(request: EvaluationRequest, *, job_id: Optional[str] = None) -> str:
    eval_jobs_root().mkdir(parents=True, exist_ok=True)
    job_id = job_id or uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    if job_dir.exists():
        raise FileExistsError(f"Evaluation job already exists: {job_id}")

    job_dir.mkdir(parents=True, exist_ok=False)
    with open(job_dir / REQUEST_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(request.model_dump(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    (job_dir / LOG_FILENAME).write_text("", encoding="utf-8")

    manifest = {
        "job_id": job_id,
        "job_kind": "evaluate",
        "status": "queued",
        "model_name": request.model_name.strip().lower(),
        "benchmark_name": request.benchmark_name,
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "device_requested": request.device or "auto",
        "device_assigned": None,
        "queue_position": None,
        "weights_id": request.weights,
        "error": None,
        "result": None,
    }
    _write_manifest(job_dir, manifest)
    return job_id


def get_job(job_id: str) -> Dict[str, Any]:
    manifest_path = _job_dir(job_id) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Unknown evaluation job: {job_id}")
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def list_jobs(*, limit: int = 50) -> List[Dict[str, Any]]:
    root = eval_jobs_root()
    if not root.is_dir():
        return []
    manifests: List[Dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue
        with open(manifest_path, encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    manifests.sort(key=lambda manifest: manifest.get("created_at", ""), reverse=True)
    return manifests[:limit]


def update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    manifest = get_job(job_id)
    manifest.update(fields)
    _write_manifest(_job_dir(job_id), manifest)
    return manifest


def mark_running(job_id: str) -> Dict[str, Any]:
    manifest = get_job(job_id)
    if manifest.get("status") == "stopping":
        raise JobCancelledError(f"Evaluation job {job_id} stop requested")
    return update_job(job_id, status="running", started_at=_utc_now_iso())


def mark_stopping(job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
    return update_job(job_id, status="stopping", error=reason)


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "status": "succeeded",
        "finished_at": _utc_now_iso(),
        "result": result,
        "error": None,
    }
    resolved_weights = result.get("weights") or result.get("weights_id")
    if resolved_weights:
        fields["weights_id"] = resolved_weights
    return update_job(job_id, **fields)


def mark_skipped(job_id: str, result: Dict[str, Any], *, reason: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "status": "skipped",
        "finished_at": _utc_now_iso(),
        "result": result,
        "error": reason,
    }
    resolved_weights = result.get("weights") or result.get("weights_id")
    if resolved_weights:
        fields["weights_id"] = resolved_weights
    return update_job(job_id, **fields)


def mark_failed(job_id: str, error: str) -> Dict[str, Any]:
    return update_job(
        job_id,
        status="failed",
        finished_at=_utc_now_iso(),
        error=error,
    )


def mark_cancelled(job_id: str, *, reason: str = "Cancelled by user") -> Dict[str, Any]:
    return update_job(
        job_id,
        status="cancelled",
        finished_at=_utc_now_iso(),
        error=reason,
        result=None,
    )


def delete_job(job_id: str) -> None:
    """Remove a job directory and all artifacts (logs, manifest, request)."""
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return
    shutil.rmtree(job_dir)


def append_log(job_id: str, message: str) -> None:
    log_path = _job_dir(job_id) / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


def read_logs(job_id: str, *, offset: int = 0) -> tuple[str, int]:
    log_path = _job_dir(job_id) / LOG_FILENAME
    if not log_path.is_file():
        return "", 0
    with open(log_path, encoding="utf-8") as handle:
        handle.seek(max(offset, 0))
        chunk = handle.read()
        next_offset = handle.tell()
    return chunk, next_offset


def wait_for_job(
    job_id: str,
    *,
    poll_interval: float = 0.5,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    deadline = time.time() + timeout if timeout is not None else None
    while True:
        manifest = get_job(job_id)
        if manifest["status"] in TERMINAL_JOB_STATUSES:
            return manifest
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
        time.sleep(poll_interval)


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
    from ..compute.job_logging import job_log_context as _job_log_context

    with _job_log_context(job_id, log_path=_job_dir(job_id) / LOG_FILENAME):
        yield

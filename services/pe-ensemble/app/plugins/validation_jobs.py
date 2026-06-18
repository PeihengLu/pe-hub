"""Filesystem-backed plugin validation job registry."""
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
from .config import validation_jobs_root

MANIFEST_FILENAME = "manifest.json"
LOG_FILENAME = "validation.log"
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled", "skipped"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _job_dir(job_id: str) -> Path:
    return validation_jobs_root() / job_id


def _write_manifest(job_dir: Path, manifest: Dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_job(plugin_name: str, *, job_id: Optional[str] = None) -> str:
    validation_jobs_root().mkdir(parents=True, exist_ok=True)
    job_id = job_id or uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    if job_dir.exists():
        raise FileExistsError(f"Validation job already exists: {job_id}")

    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / LOG_FILENAME).write_text("", encoding="utf-8")

    manifest = {
        "job_id": job_id,
        "job_kind": "plugin_validate",
        "status": "queued",
        "plugin_name": plugin_name.strip().lower(),
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
    }
    _write_manifest(job_dir, manifest)
    return job_id


def get_job(job_id: str) -> Dict[str, Any]:
    manifest_path = _job_dir(job_id) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Unknown plugin validation job: {job_id}")
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def list_jobs_for_plugin(plugin_name: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    key = plugin_name.strip().lower()
    root = validation_jobs_root()
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
            manifest = json.load(handle)
        if manifest.get("plugin_name") == key:
            manifests.append(manifest)
    manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return manifests[:limit]


def find_active_job(plugin_name: str) -> Optional[Dict[str, Any]]:
    for manifest in list_jobs_for_plugin(plugin_name, limit=20):
        if manifest.get("status") in ("queued", "running", "stopping"):
            return manifest
    return None


def update_job(job_id: str, **fields: Any) -> Dict[str, Any]:
    manifest = get_job(job_id)
    manifest.update(fields)
    _write_manifest(_job_dir(job_id), manifest)
    return manifest


def mark_running(job_id: str) -> Dict[str, Any]:
    manifest = get_job(job_id)
    if manifest.get("status") == "stopping":
        raise JobCancelledError(f"Validation job {job_id} stop requested")
    return update_job(job_id, status="running", started_at=_utc_now_iso())


def mark_stopping(job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
    return update_job(job_id, status="stopping", error=reason)


def mark_succeeded(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return update_job(
        job_id,
        status="succeeded",
        finished_at=_utc_now_iso(),
        result=result,
        error=None,
    )


def mark_failed(job_id: str, error: str) -> Dict[str, Any]:
    return update_job(
        job_id,
        status="failed",
        finished_at=_utc_now_iso(),
        error=error,
        result=None,
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
    job_dir = _job_dir(job_id)
    if job_dir.exists():
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
            raise TimeoutError(f"Validation job {job_id} did not finish within {timeout}s")
        time.sleep(poll_interval)


@contextmanager
def job_log_context(job_id: str):
    from ..compute.job_logging import job_log_context as _job_log_context

    with _job_log_context(job_id, log_path=_job_dir(job_id) / LOG_FILENAME):
        yield

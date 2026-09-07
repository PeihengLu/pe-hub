"""Shared filesystem job store for train/tune/eval/ensemble/plugin validation."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .job_cancel import JobCancelledError
from .manifest_io import read_json_retry, write_json_atomic

MANIFEST_FILENAME = "manifest.json"
DEFAULT_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def utc_now_iso() -> str:
    # Millisecond precision, not whole seconds: job listings sort on this
    # string, and two jobs submitted in the same second would otherwise tie
    # and come back in arbitrary directory order.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JobStore:
    root: Callable[[], Path]
    log_filename: str
    unknown_message: str
    request_filename: Optional[str] = "request.json"
    terminal_statuses: frozenset[str] = DEFAULT_TERMINAL

    def job_dir(self, job_id: str) -> Path:
        return self.root() / job_id

    def create(
        self,
        manifest: Dict[str, Any],
        *,
        job_id: Optional[str] = None,
        request_payload: Optional[Dict[str, Any]] = None,
        exists_message: str,
    ) -> str:
        self.root().mkdir(parents=True, exist_ok=True)
        job_id = job_id or uuid.uuid4().hex
        job_dir = self.job_dir(job_id)
        if job_dir.exists():
            raise FileExistsError(exists_message.format(job_id=job_id))
        job_dir.mkdir(parents=True, exist_ok=False)
        if self.request_filename is not None and request_payload is not None:
            with open(job_dir / self.request_filename, "w", encoding="utf-8") as handle:
                json.dump(request_payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        (job_dir / self.log_filename).write_text("", encoding="utf-8")
        payload = {"job_id": job_id, **manifest}
        payload.setdefault("created_at", utc_now_iso())
        write_json_atomic(job_dir / MANIFEST_FILENAME, payload)
        return job_id

    def get(self, job_id: str) -> Dict[str, Any]:
        return read_json_retry(
            self.job_dir(job_id) / MANIFEST_FILENAME,
            missing_message=self.unknown_message.format(job_id=job_id),
        )

    def list(
        self,
        *,
        limit: int = 50,
        match: Callable[[Dict[str, Any]], bool] | None = None,
    ) -> list[Dict[str, Any]]:
        root = self.root()
        if not root.is_dir():
            return []
        manifests: list[Dict[str, Any]] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            manifest_path = entry / MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            manifests.append(
                read_json_retry(
                    manifest_path,
                    missing_message=self.unknown_message.format(job_id=entry.name),
                )
            )
        if match is not None:
            manifests = [item for item in manifests if match(item)]
        manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return manifests[:limit]

    def request_path(self, job_id: str) -> Path:
        if self.request_filename is None:
            raise ValueError("This job store has no request file")
        return self.job_dir(job_id) / self.request_filename

    def load_request_json(self, job_id: str) -> Dict[str, Any]:
        with open(self.request_path(job_id), encoding="utf-8") as handle:
            return json.load(handle)

    def update(self, job_id: str, **fields: Any) -> Dict[str, Any]:
        manifest = self.get(job_id)
        manifest.update(fields)
        write_json_atomic(self.job_dir(job_id) / MANIFEST_FILENAME, manifest)
        return manifest

    def mark_running(self, job_id: str, *, kind_label: str) -> Dict[str, Any]:
        manifest = self.get(job_id)
        if manifest.get("status") == "stopping":
            raise JobCancelledError(f"{kind_label} job {job_id} stop requested")
        return self.update(job_id, status="running", started_at=utc_now_iso())

    def mark_stopping(self, job_id: str, *, reason: str = "Stop requested") -> Dict[str, Any]:
        return self.update(job_id, status="stopping", error=reason)

    def mark_cancelled(
        self, job_id: str, *, reason: str = "Cancelled by user"
    ) -> Dict[str, Any]:
        return self.update(
            job_id,
            status="cancelled",
            finished_at=utc_now_iso(),
            error=reason,
            result=None,
        )

    def mark_failed(
        self,
        job_id: str,
        error: str,
        **fields: Any,
    ) -> Dict[str, Any]:
        payload = {
            "status": "failed",
            "finished_at": utc_now_iso(),
            "error": error,
            **fields,
        }
        return self.update(job_id, **payload)

    def mark_terminal(self, job_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        return self.update(job_id, status=status, finished_at=utc_now_iso(), **fields)

    def delete(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)

    def append_log(self, job_id: str, message: str) -> None:
        log_path = self.job_dir(job_id) / self.log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip("\n") + "\n")

    def read_logs(self, job_id: str, *, offset: int = 0) -> tuple[str, int]:
        log_path = self.job_dir(job_id) / self.log_filename
        if not log_path.is_file():
            return "", 0
        with open(log_path, encoding="utf-8") as handle:
            handle.seek(max(offset, 0))
            chunk = handle.read()
            return chunk, handle.tell()

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 0.5,
        timeout: Optional[float] = None,
        timeout_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            manifest = self.get(job_id)
            if manifest["status"] in self.terminal_statuses:
                return manifest
            if deadline is not None and time.time() >= deadline:
                message = timeout_message or f"Job {job_id} did not finish within {timeout}s"
                raise TimeoutError(message)
            time.sleep(poll_interval)

    @contextmanager
    def job_log_context(self, job_id: str):
        from .job_logging import job_log_context as _job_log_context

        with _job_log_context(job_id, log_path=self.job_dir(job_id) / self.log_filename):
            yield

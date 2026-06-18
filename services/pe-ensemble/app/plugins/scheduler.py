"""Background scheduler for plugin validation jobs (CPU, single worker)."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from pe_common.plugins import PluginError

from ..compute.job_cancel import clear_cancel, register_cancel_event, request_cancel
from .runner import run_plugin_validation_job
from .validation_jobs import create_job, find_active_job, get_job, mark_cancelled

logger = logging.getLogger(__name__)


class PluginValidationScheduler:
    """Queue plugin validation work off the API event loop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pe-validate")
        self._plugin_job: Dict[str, str] = {}

    def submit(self, plugin_name: str) -> str:
        key = plugin_name.strip().lower()
        with self._lock:
            active = find_active_job(key)
            if active is not None:
                raise PluginError(
                    f"Validation already in progress for '{key}' (job {active['job_id']})"
                )
            job_id = create_job(key)
            self._plugin_job[key] = job_id
            register_cancel_event("validate", job_id)
            self._executor.submit(self._run, key, job_id)
            return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            try:
                manifest = get_job(job_id)
            except FileNotFoundError:
                return False
            if manifest.get("status") not in ("queued", "running", "stopping"):
                return False
            request_cancel("validate", job_id)
            if manifest.get("status") == "queued":
                mark_cancelled(job_id)
                plugin_name = manifest.get("plugin_name")
                if plugin_name and self._plugin_job.get(plugin_name) == job_id:
                    del self._plugin_job[plugin_name]
                return True
            return True

    def _run(self, plugin_name: str, job_id: str) -> None:
        try:
            run_plugin_validation_job(job_id)
        finally:
            clear_cancel("validate", job_id)
            with self._lock:
                if self._plugin_job.get(plugin_name) == job_id:
                    del self._plugin_job[plugin_name]

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


_scheduler: Optional[PluginValidationScheduler] = None
_scheduler_lock = threading.Lock()


def get_validation_scheduler() -> PluginValidationScheduler:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = PluginValidationScheduler()
        return _scheduler


def shutdown_validation_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            _scheduler = None

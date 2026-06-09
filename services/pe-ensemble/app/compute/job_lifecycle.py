"""Kill running or queued jobs and remove their on-disk artifacts."""
from __future__ import annotations

import time
from typing import Callable, Dict

from .device_scheduler import JobKind, get_scheduler
from .job_cancel import clear_cancel

KILL_WAIT_TIMEOUT = 60.0
KILL_POLL_INTERVAL = 0.2
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def _wait_for_terminal(get_job: Callable[[str], Dict[str, object]], job_id: str) -> None:
    deadline = time.time() + KILL_WAIT_TIMEOUT
    while time.time() < deadline:
        try:
            manifest = get_job(job_id)
        except FileNotFoundError:
            return
        if manifest.get("status") in TERMINAL_STATUSES:
            return
        time.sleep(KILL_POLL_INTERVAL)


def kill_and_remove_job(
    kind: JobKind,
    job_id: str,
    *,
    get_job: Callable[[str], Dict[str, object]],
    delete_job: Callable[[str], None],
) -> None:
    """Cancel if active, wait briefly for a running worker, then delete job files."""
    try:
        manifest = get_job(job_id)
    except FileNotFoundError:
        delete_job(job_id)
        clear_cancel(kind, job_id)
        return

    status = manifest.get("status")
    scheduler = get_scheduler()

    if status in ("queued", "running"):
        scheduler.cancel_job(kind, job_id)

    if status == "running":
        _wait_for_terminal(get_job, job_id)

    delete_job(job_id)
    clear_cancel(kind, job_id)

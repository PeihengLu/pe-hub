"""Kill running or queued jobs and remove their on-disk artifacts."""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from .device_scheduler import JobKind, get_scheduler
from .job_cancel import clear_cancel

KILL_WAIT_TIMEOUT = 60.0
KILL_POLL_INTERVAL = 0.2
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "skipped"})
ACTIVE_KILL_STATUSES = frozenset({"queued", "running", "stopping"})


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


def begin_job_kill(
    kind: JobKind,
    job_id: str,
    *,
    get_job: Callable[[str], Dict[str, object]],
) -> Optional[Dict[str, object]]:
    """Request cancellation and mark active jobs as stopping (fast, non-blocking)."""
    try:
        manifest = get_job(job_id)
    except FileNotFoundError:
        return None

    if manifest.get("status") in ACTIVE_KILL_STATUSES:
        get_scheduler().cancel_job(kind, job_id)
        try:
            return get_job(job_id)
        except FileNotFoundError:
            return None
    return manifest


def finalize_job_kill(
    kind: JobKind,
    job_id: str,
    *,
    get_job: Callable[[str], Dict[str, object]],
    delete_job: Callable[[str], None],
) -> None:
    """Wait briefly for a worker to stop, then delete job files."""
    try:
        manifest = get_job(job_id)
    except FileNotFoundError:
        delete_job(job_id)
        clear_cancel(kind, job_id)
        return

    if manifest.get("status") in ("running", "stopping"):
        _wait_for_terminal(get_job, job_id)

    delete_job(job_id)
    clear_cancel(kind, job_id)


def kill_and_remove_job(
    kind: JobKind,
    job_id: str,
    *,
    get_job: Callable[[str], Dict[str, object]],
    delete_job: Callable[[str], None],
) -> None:
    """Cancel if active, wait briefly for a running worker, then delete job files."""
    begin_job_kill(kind, job_id, get_job=get_job)
    finalize_job_kill(kind, job_id, get_job=get_job, delete_job=delete_job)

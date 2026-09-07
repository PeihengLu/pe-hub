"""Cooperative cancellation flags for queued and running compute jobs."""
from __future__ import annotations

import threading
from typing import Dict, Tuple

JobKey = Tuple[str, str]

_cancel_events: Dict[JobKey, threading.Event] = {}
_lock = threading.Lock()


def register_cancel_event(kind: str, job_id: str) -> threading.Event:
    key = (kind, job_id)
    with _lock:
        event = threading.Event()
        _cancel_events[key] = event
        return event


def request_cancel(kind: str, job_id: str) -> None:
    key = (kind, job_id)
    with _lock:
        event = _cancel_events.get(key)
        if event is not None:
            event.set()


def is_cancel_requested(kind: str, job_id: str) -> bool:
    key = (kind, job_id)
    with _lock:
        event = _cancel_events.get(key)
        return bool(event and event.is_set())


def clear_cancel(kind: str, job_id: str) -> None:
    key = (kind, job_id)
    with _lock:
        _cancel_events.pop(key, None)


class JobCancelledError(Exception):
    """Raised when a job is stopped by the user."""

"""Tail PE-DB conversion progress into ensemble job logs."""
from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from pe_common.conversion_progress import clear_progress, progress_file_path

from .config import use_pe_db_library


def _tail_progress_file(
    path,
    log_fn: Callable[[str], None],
    stop_event: threading.Event,
) -> None:
    offset = 0
    while not stop_event.is_set():
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            for line in chunk.splitlines():
                text = line.strip()
                if text:
                    log_fn(text)
        stop_event.wait(0.25)

    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
        for line in chunk.splitlines():
            text = line.strip()
            if text:
                log_fn(text)


@contextmanager
def pe_db_filter_progress(
    params: Dict[str, Any],
    *,
    progress_log: Optional[Callable[[str], None]],
    request_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Call PE-DB filter/export and mirror coarse conversion progress into job logs."""
    if progress_log is None or use_pe_db_library():
        # CLI already streams conversion progress via progress_callback.
        yield request_fn(params)
        return

    token = uuid.uuid4().hex
    path = progress_file_path(token)
    clear_progress(token)
    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_progress_file,
        args=(path, progress_log, stop_event),
        daemon=True,
        name="pe-db-progress-tail",
    )
    tail_thread.start()
    try:
        yield request_fn({**params, "progress_token": token})
    finally:
        stop_event.set()
        tail_thread.join(timeout=2.0)
        clear_progress(token)

"""Per-job log capture safe for concurrent compute workers."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

_job_context = threading.local()


def _current_job_id() -> str | None:
    return getattr(_job_context, "job_id", None)


@contextmanager
def job_log_context(job_id: str, *, log_path: Path) -> None:
    """Capture logging for the current worker thread into a job log file."""
    previous_job_id = _current_job_id()
    _job_context.job_id = job_id

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    class _JobLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return _current_job_id() == job_id

    class _JobLogHandler(logging.Handler):
        def __init__(self, append: Callable[[str], None]) -> None:
            super().__init__()
            self._append = append

        def emit(self, record: logging.LogRecord) -> None:
            try:
                self._append(self.format(record))
            except Exception:  # noqa: BLE001
                self.handleError(record)

    def _append(message: str) -> None:
        log_file.write(message.rstrip("\n") + "\n")
        log_file.flush()

    handler = _JobLogHandler(_append)
    handler.addFilter(_JobLogFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
        log_file.close()
        if previous_job_id is None:
            if hasattr(_job_context, "job_id"):
                delattr(_job_context, "job_id")
        else:
            _job_context.job_id = previous_job_id

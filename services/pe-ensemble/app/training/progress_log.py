"""Helpers for streaming training progress into ensemble job logs."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from pe_common.training import format_epoch_metrics_row

JOB_PROGRESS_LOG_KEY = "_job_progress_log"
JOB_CANCEL_CHECK_KEY = "_job_cancel_check"
ProgressLog = Callable[[str], None]
CancelCheck = Callable[[], None]


class _DiscardStream:
    """Sink that swallows all writes (used to silence tqdm on stderr)."""

    def write(self, data: str) -> int:
        return len(data) if data else 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class _LineTee:
    """Mirror stream writes into the job log."""

    def __init__(
        self,
        underlying: Any,
        append: Optional[ProgressLog],
        *,
        cancel_check: Optional[CancelCheck] = None,
    ) -> None:
        self._underlying = underlying
        self._append = append
        self._cancel_check = cancel_check
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        if self._cancel_check is not None:
            self._cancel_check()
        self._underlying.write(data)
        if self._append is None:
            return len(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._append(stripped)
        return len(data)

    def flush(self) -> None:
        self._underlying.flush()
        if self._append is not None and self._buffer.strip():
            self._append(self._buffer.strip())
            self._buffer = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


def take_job_training_callbacks(
    hyperparameters: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], Optional[ProgressLog], Optional[CancelCheck]]:
    """Return hyperparameters without internal job callbacks."""
    params = dict(hyperparameters or {})
    progress_log = params.pop(JOB_PROGRESS_LOG_KEY, None)
    cancel_check = params.pop(JOB_CANCEL_CHECK_KEY, None)
    if progress_log is not None and not callable(progress_log):
        progress_log = None
    if cancel_check is not None and not callable(cancel_check):
        cancel_check = None
    return params, progress_log, cancel_check


def take_job_progress_log(
    hyperparameters: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], Optional[ProgressLog]]:
    """Return a copy of hyperparameters without the internal job-log callback."""
    params, progress_log, _cancel_check = take_job_training_callbacks(hyperparameters)
    return params, progress_log


def make_epoch_logger(
    progress_log: Optional[ProgressLog],
    *,
    prefix: str = "",
    cancel_check: Optional[CancelCheck] = None,
) -> Optional[Callable[[Dict[str, float]], None]]:
    if progress_log is None and cancel_check is None:
        return None

    def _log(row: Dict[str, float]) -> None:
        if cancel_check is not None:
            cancel_check()
        if progress_log is not None:
            progress_log(format_epoch_metrics_row(row, prefix=prefix))

    return _log


def log_training_best(
    progress_log: Optional[ProgressLog],
    *,
    best_epoch: int,
    best_val_loss: float,
    prefix: str = "",
) -> None:
    if progress_log is None or best_epoch < 0:
        return
    label = f"{prefix.rstrip()} | " if prefix else ""
    progress_log(f"{label}best epoch {best_epoch} | val_loss={best_val_loss:.6g}")


@contextmanager
def tee_stream_to_log(
    progress_log: Optional[ProgressLog],
    *,
    stream: Any = None,
    stderr: bool = False,
    cancel_check: Optional[CancelCheck] = None,
) -> Iterator[None]:
    """Mirror vendor stdout into the job log; optionally silence tqdm on stderr."""
    if progress_log is None and cancel_check is None and not stderr:
        yield
        return

    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    stdout_tee: Optional[_LineTee] = None
    if progress_log is not None or cancel_check is not None:
        stdout_tee = _LineTee(
            stream or sys.stdout,
            progress_log,
            cancel_check=cancel_check,
        )
        sys.stdout = stdout_tee
    if stderr:
        sys.stderr = _DiscardStream()
    try:
        yield
    finally:
        if stdout_tee is not None:
            stdout_tee.flush()
            sys.stdout = previous_stdout
        if stderr:
            sys.stderr = previous_stderr

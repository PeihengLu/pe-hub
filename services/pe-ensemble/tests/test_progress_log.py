"""Tests for training job progress logging helpers."""
from __future__ import annotations

import io
import sys

from app.training.progress_log import (
    JOB_CANCEL_CHECK_KEY,
    JOB_PROGRESS_LOG_KEY,
    make_epoch_logger,
    take_job_progress_log,
    take_job_training_callbacks,
    tee_stream_to_log,
)


def test_take_job_progress_log_removes_internal_key():
    messages: list[str] = []

    def _log(message: str) -> None:
        messages.append(message)

    params, callback = take_job_progress_log(
        {"epochs": 5, JOB_PROGRESS_LOG_KEY: _log},
    )
    assert params == {"epochs": 5}
    assert callback is _log


def test_take_job_training_callbacks_removes_internal_keys():
    messages: list[str] = []
    checks: list[str] = []

    def _log(message: str) -> None:
        messages.append(message)

    def _check() -> None:
        checks.append("checked")

    params, progress_log, cancel_check = take_job_training_callbacks(
        {"epochs": 5, JOB_PROGRESS_LOG_KEY: _log, JOB_CANCEL_CHECK_KEY: _check},
    )
    assert params == {"epochs": 5}
    assert progress_log is _log
    assert cancel_check is _check


def test_make_epoch_logger_formats_metrics():
    lines: list[str] = []
    logger = make_epoch_logger(lines.append, prefix="fold_0 |")
    assert logger is not None
    logger({"epoch": 1.0, "train_loss": 0.2, "val_loss": 0.3})
    assert lines == ["fold_0 | epoch 1 | train_loss=0.2 | val_loss=0.3"]


def test_make_epoch_logger_runs_cancel_check():
    checks: list[str] = []

    def _check() -> None:
        checks.append("checked")

    logger = make_epoch_logger(None, cancel_check=_check)
    assert logger is not None
    logger({"epoch": 1.0, "train_loss": 0.2, "val_loss": 0.3})
    assert checks == ["checked"]


def test_tee_stream_to_log_captures_stdout_lines():
    lines: list[str] = []
    buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer
    try:
        with tee_stream_to_log(lines.append, stream=buffer):
            print("epoch 0 | train_loss=0.1")
            print("epoch 1 | train_loss=0.05")
    finally:
        sys.stdout = original_stdout
    assert lines == [
        "epoch 0 | train_loss=0.1",
        "epoch 1 | train_loss=0.05",
    ]


def test_tee_stream_to_log_suppresses_tqdm_stderr():
    lines: list[str] = []
    buffer = io.StringIO()
    original_stderr = sys.stderr
    sys.stderr = buffer
    try:
        with tee_stream_to_log(lines.append, stderr=True):
            sys.stderr.write("\raligning sequences:  50%|████     | 5/10 [00:01<00:01, 5.00it/s]")
            sys.stderr.write("\raligning sequences: 100%|████| 10/10 [00:01<00:00, 9.00it/s]")
            sys.stderr.flush()
            print("prediction started")
    finally:
        sys.stderr = original_stderr
    assert buffer.getvalue() == ""
    assert lines == ["prediction started"]
    assert "aligning" not in "".join(lines)

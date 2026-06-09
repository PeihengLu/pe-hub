"""Evaluation job configuration."""
from __future__ import annotations

import os
from pathlib import Path


def eval_jobs_root() -> Path:
    env = os.getenv("EVAL_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "eval_jobs").resolve()

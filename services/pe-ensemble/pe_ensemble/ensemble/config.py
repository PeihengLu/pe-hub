"""Ensemble job configuration."""
from __future__ import annotations

import os
from pathlib import Path


def ensemble_jobs_root() -> Path:
    env = os.getenv("ENSEMBLE_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "ensemble_jobs").resolve()

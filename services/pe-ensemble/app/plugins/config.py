"""Plugin validation job configuration."""
from __future__ import annotations

import os
from pathlib import Path


def validation_jobs_root() -> Path:
    env = os.getenv("PLUGIN_VALIDATION_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "validation_jobs").resolve()

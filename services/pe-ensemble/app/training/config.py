"""Training service configuration."""
from __future__ import annotations

import os
from pathlib import Path

SUPPORTED_MODELS = ("deepprime", "pridict2", "oped")

MODEL_FORMAT = {
    "deepprime": "deepprime",
    "pridict2": "pridict2",
    "oped": "oped",
}


def pe_db_url() -> str:
    return os.getenv("PE_DB_URL", "http://localhost:8000")


def jobs_root() -> Path:
    env = os.getenv("TRAINING_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "jobs").resolve()

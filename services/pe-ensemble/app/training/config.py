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


def pe_db_filter_timeout() -> tuple[float, float]:
    """Return ``(connect_timeout, read_timeout)`` for PE-DB ``/api/filter`` requests."""
    connect = float(os.getenv("PE_DB_FILTER_CONNECT_TIMEOUT_SECONDS", "10"))
    read = float(os.getenv("PE_DB_FILTER_TIMEOUT_SECONDS", "600"))
    return connect, read


def jobs_root() -> Path:
    env = os.getenv("TRAINING_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "jobs").resolve()

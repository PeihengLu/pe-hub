"""Training service configuration."""
from __future__ import annotations

import os
from pathlib import Path

from ..models.registry import model_registry


def supported_models() -> tuple[str, ...]:
    return model_registry.names()


def is_supported_model(model_name: str) -> bool:
    return model_registry.is_registered(model_name)


def model_format_for(model_name: str) -> str:
    return model_registry.get(model_name).pe_db_format


def model_format_map() -> dict[str, str]:
    return model_registry.model_format_map()


# Backward-compatible module-level aliases (derived from the registry).
SUPPORTED_MODELS = supported_models()
MODEL_FORMAT = model_format_map()


def pe_db_url() -> str:
    return os.getenv("PE_DB_URL", "http://localhost:8000")


def pe_db_mode() -> str:
    """``http`` (default) or ``library`` for in-process PE-DB access."""
    return os.getenv("PE_DB_MODE", "http").strip().lower()


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

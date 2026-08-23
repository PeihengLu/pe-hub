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


# Process flag: CLI enables in-process pe-db library; FastAPI server leaves this False (HTTP).
_use_pe_db_library: bool = False


def enable_cli_pe_db_access() -> None:
    """Use in-process ``pe_db.library`` (same code path as the ``pe-db`` CLI).

    Called once from the ``pe-ensemble`` CLI entrypoint. The FastAPI service never
    calls this, so it always talks to PE-DB over HTTP via ``PE_DB_URL``.
    """
    global _use_pe_db_library
    _use_pe_db_library = True


def use_pe_db_library() -> bool:
    """True when pe-ensemble is running as a CLI (in-process PE-DB)."""
    return _use_pe_db_library


def pe_db_mode() -> str:
    """Return ``library`` for the pe-ensemble CLI or ``http`` for the web service.

    Prefer :func:`use_pe_db_library`. This string form exists only as a thin alias.
    """
    return "library" if _use_pe_db_library else "http"


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


def presets_root() -> Path:
    env = os.getenv("TRAINING_PRESETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "config" / "training_presets").resolve()


def tuning_studies_root() -> Path:
    env = os.getenv("TUNING_STUDIES_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "tuning_studies").resolve()


def tune_jobs_root() -> Path:
    env = os.getenv("TUNING_JOBS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "tune_jobs").resolve()

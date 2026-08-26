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

# Collision-safe alias used by ``pe_ensemble.library`` (see pe_ensemble._bootstrap).
_SERVICE_APP_CONFIG = "pe_ensemble_service_app.training.config"
_APP_CONFIG = "app.training.config"


def _sync_pe_db_library_flag(enabled: bool) -> None:
    """Keep both ``app`` and ``pe_ensemble_service_app`` copies of this module in sync.

    The CLI may import ``app.training.*`` while ``pe_ensemble.library`` loads the
    same files under ``pe_ensemble_service_app.*``. A process-global flag must be
    mirrored or evaluate/ensemble --sync would still talk HTTP to PE_DB_URL.
    """
    import sys

    for name in (_APP_CONFIG, _SERVICE_APP_CONFIG, __name__):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if getattr(mod, "__file__", None) != __file__:
            continue
        mod._use_pe_db_library = enabled


def enable_cli_pe_db_access() -> None:
    """Use in-process ``pe_db.library`` (same code path as the ``pe-db`` CLI).

    Called once from the ``pe-ensemble`` CLI entrypoint. The FastAPI service never
    calls this, so it always talks to PE-DB over HTTP via ``PE_DB_URL``.
    """
    _sync_pe_db_library_flag(True)


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


def shipped_presets_root() -> Path:
    """Git-tracked shared defaults (``config/training_presets``).

    Override with ``TRAINING_SHIPPED_PRESETS_ROOT`` (tests / unusual layouts).
    """
    env = os.getenv("TRAINING_SHIPPED_PRESETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "config" / "training_presets").resolve()


def local_presets_root() -> Path:
    """Writable HPO overlay (``config/training_presets_local``, gitignored).

    ``TRAINING_PRESETS_ROOT`` (or ``TRAINING_LOCAL_PRESETS_ROOT``) overrides this.
    Tune writes dataset entries here; train merges local over shipped.
    """
    env = os.getenv("TRAINING_PRESETS_ROOT") or os.getenv("TRAINING_LOCAL_PRESETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2] / "config" / "training_presets_local"
    ).resolve()


def presets_root() -> Path:
    """Writable presets root (local overlay). Prefer :func:`local_presets_root`."""
    return local_presets_root()


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

"""PE-DB data access for PE Ensemble.

CLI runs use in-process ``pe_db.library`` (same path as the ``pe-db`` CLI).
The FastAPI web service always uses HTTP against ``PE_DB_URL``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import requests

from .config import pe_db_filter_timeout, pe_db_url, use_pe_db_library

ProgressCallback = Callable[[str], None]


class PeDbAccessError(RuntimeError):
    """Raised when PE-DB filter/export fails."""


def _fetch_via_http(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(
        f"{pe_db_url()}/api/filter",
        params=params,
        timeout=pe_db_filter_timeout(),
    )
    response.raise_for_status()
    return response.json()


def _fetch_via_library(
    params: Dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    try:
        from pe_db.library import filter_from_params
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "pe_db" or missing.startswith("pe_db."):
            raise PeDbAccessError(
                "CLI PE-DB access requires the pe-db package. "
                "Install with: pip install -e services/pe-db"
            ) from exc
        raise PeDbAccessError(
            f"PE-DB library import failed (missing dependency: {missing}). "
            "Install pe-db and its dependencies with: pip install -e services/pe-db"
        ) from exc
    except ImportError as exc:
        raise PeDbAccessError(
            "Could not import pe-db for CLI access. "
            "Install with: pip install -e services/pe-db"
        ) from exc
    return filter_from_params(params, progress_callback=progress_callback)


def fetch_pe_db_filter(
    params: Dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Fetch filtered/model-format data (in-process ``pe_db.library`` in CLI, HTTP in the web service)."""
    if use_pe_db_library():
        return _fetch_via_library(params, progress_callback=progress_callback)
    return _fetch_via_http(params)


def reload_pe_db_plugins() -> Dict[str, Any]:
    """Reload PE-DB plugin converters after ensemble plugin activation."""
    if use_pe_db_library():
        try:
            from pe_db.library import reload_plugins
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "pe_db" or missing.startswith("pe_db."):
                raise PeDbAccessError(
                    "CLI PE-DB access requires the pe-db package. "
                    "Install with: pip install -e services/pe-db"
                ) from exc
            raise PeDbAccessError(
                f"PE-DB library import failed (missing dependency: {missing}). "
                "Install pe-db and its dependencies with: pip install -e services/pe-db"
            ) from exc
        except ImportError as exc:
            raise PeDbAccessError(
                "Could not import pe-db for CLI access. "
                "Install with: pip install -e services/pe-db"
            ) from exc
        loaded = reload_plugins()
        return {"loaded": loaded, "count": len(loaded)}
    url = f"{pe_db_url().rstrip('/')}/api/plugins/reload"
    try:
        response = requests.post(url, timeout=(10, 120))
    except requests.RequestException as exc:
        raise PeDbAccessError(f"PE-DB plugin reload request failed: {exc}") from exc
    if response.status_code >= 400:
        raise PeDbAccessError(
            f"PE-DB plugin reload failed ({response.status_code}): {response.text}"
        )
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}

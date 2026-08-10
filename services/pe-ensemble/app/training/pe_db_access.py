"""PE-DB data access for PE Ensemble (HTTP or in-process library mode)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import requests

from .config import pe_db_filter_timeout, pe_db_mode, pe_db_url

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
                "PE_DB_MODE=library requires the pe-db package. "
                "Install with: pip install -e services/pe-db"
            ) from exc
        raise PeDbAccessError(
            f"PE-DB library import failed (missing dependency: {missing}). "
            "Install pe-db and its dependencies with: pip install -e services/pe-db"
        ) from exc
    except ImportError as exc:
        raise PeDbAccessError(
            "PE_DB_MODE=library could not import pe-db. "
            "Install with: pip install -e services/pe-db"
        ) from exc
    return filter_from_params(params, progress_callback=progress_callback)


def fetch_pe_db_filter(
    params: Dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Fetch filtered/model-format data using the configured PE-DB transport."""
    if pe_db_mode() == "library":
        return _fetch_via_library(params, progress_callback=progress_callback)
    return _fetch_via_http(params)


def reload_pe_db_plugins() -> Dict[str, Any]:
    """Reload PE-DB plugin converters after ensemble plugin activation."""
    if pe_db_mode() == "library":
        try:
            from pe_db.library import reload_plugins
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "pe_db" or missing.startswith("pe_db."):
                raise PeDbAccessError(
                    "PE_DB_MODE=library requires the pe-db package. "
                    "Install with: pip install -e services/pe-db"
                ) from exc
            raise PeDbAccessError(
                f"PE-DB library import failed (missing dependency: {missing}). "
                "Install pe-db and its dependencies with: pip install -e services/pe-db"
            ) from exc
        except ImportError as exc:
            raise PeDbAccessError(
                "PE_DB_MODE=library could not import pe-db. "
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

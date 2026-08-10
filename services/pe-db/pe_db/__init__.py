"""Installable PE Database library package (CLI + in-process access).

The FastAPI service lives under ``app/``; this namespace is the stable import
path for headless use (``PE_DB_MODE=library`` in PE Ensemble).
"""
from __future__ import annotations

from pe_db.library import (
    PeDbLibraryError,
    ensure_plugins_loaded,
    filter_data,
    filter_from_params,
    list_output_formats,
    reload_plugins,
    run_convert_sheet,
    run_export,
    run_init,
    run_seed,
    run_standardize,
)

__all__ = [
    "PeDbLibraryError",
    "ensure_plugins_loaded",
    "filter_data",
    "filter_from_params",
    "list_output_formats",
    "reload_plugins",
    "run_convert_sheet",
    "run_export",
    "run_init",
    "run_seed",
    "run_standardize",
]

__version__ = "0.2.0"

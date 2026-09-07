"""Installable PE Database package (FastAPI service, CLI, and in-process library).

``pe_db.library`` is the stable import path for headless use (``pedb`` and
in-process ``peen``). The HTTP app is ``pe_db.main:app``.
"""
from __future__ import annotations

from pe_db.library import (
    PeDbLibraryError,
    catalog_statistics,
    ensure_plugins_loaded,
    filter_data,
    filter_from_params,
    list_datasheets,
    list_datasets,
    list_output_formats,
    list_scaffolds,
    list_studies,
    reload_plugins,
    run_clear_cached_data,
    run_convert_sheet,
    run_export,
    run_init,
    run_seed,
    run_standardize,
)

__all__ = [
    "PeDbLibraryError",
    "catalog_statistics",
    "ensure_plugins_loaded",
    "filter_data",
    "filter_from_params",
    "list_datasheets",
    "list_datasets",
    "list_output_formats",
    "list_scaffolds",
    "list_studies",
    "reload_plugins",
    "run_clear_cached_data",
    "run_convert_sheet",
    "run_export",
    "run_init",
    "run_seed",
    "run_standardize",
]

__version__ = "0.2.0"

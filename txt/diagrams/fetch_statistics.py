"""Fetch PE-DB statistics from the FastAPI service and return pandas DataFrames.

Example
-------
>>> from fetch_statistics import load_statistics_tables
>>> tables = load_statistics_tables()
>>> tables["edit_type"]

Run from the command line (PE-DB API must be running on port 8000 by default)::

    python fetch_statistics.py
    python fetch_statistics.py --edit-scope on_target --save-dir ./stats_csv
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import pandas as pd
import requests

DEFAULT_BASE_URL = "http://localhost:8000"
STATISTICS_PATH = "/api/statistics"

TABLE_KEYS = (
    "edit_type",
    "edit_length",
    "pegRNA_delivery_method",
    "pe_delivery_method",
    "edit_scope",
    "experimental_method",
    "target_context",
)


def fetch_statistics(
    base_url: str = DEFAULT_BASE_URL,
    *,
    edit_type: Optional[str] = None,
    edit_length: Optional[int] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[str] = None,
    experimental_method: Optional[str] = None,
    target_context: Optional[str] = None,
    scaffold_name: Optional[str] = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call ``GET /api/statistics`` and return the JSON payload."""
    params = {
        "edit_type": edit_type,
        "edit_length": edit_length,
        "edit_efficiency_min": edit_efficiency_min,
        "edit_efficiency_max": edit_efficiency_max,
        "edit_scope": edit_scope,
        "experimental_method": experimental_method,
        "target_context": target_context,
        "scaffold_name": scaffold_name,
    }
    params = {key: value for key, value in params.items() if value is not None}

    url = base_url.rstrip("/") + STATISTICS_PATH
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def statistics_to_dataframes(payload: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Convert a statistics API payload into named DataFrames for plotting."""
    tables: dict[str, pd.DataFrame] = {}

    for key in TABLE_KEYS:
        rows = payload.get(key, [])
        tables[key] = pd.DataFrame(rows)

    tables["summary"] = pd.DataFrame(
        [
            {
                "total_entries": payload.get("total_entries", 0),
                "total_studies": payload.get("total_studies", 0),
            }
        ]
    )
    return tables


def load_statistics_tables(
    base_url: str = DEFAULT_BASE_URL,
    **filters: Any,
) -> dict[str, pd.DataFrame]:
    """Fetch statistics and return DataFrames ready for illustration."""
    payload = fetch_statistics(base_url, **filters)
    return statistics_to_dataframes(payload)
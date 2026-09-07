"""Helpers for serializing tabular data to JSON-safe Python structures."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def json_safe_value(value: Any) -> Any:
    """Convert a scalar to a value that stdlib ``json`` can encode. Cast Nan/Inf to None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return dataframe rows as JSON-serializable dicts (NaN/Inf → null)."""
    sanitized = df.replace([np.inf, -np.inf], np.nan)
    records = sanitized.to_dict(orient="records")
    return [
        {column: json_safe_value(value) for column, value in record.items()}
        for record in records
    ]

"""Standardized PE-DB rows -> native columns for my_model."""
from __future__ import annotations

import pandas as pd


def convert(std_df: pd.DataFrame) -> pd.DataFrame:
    """Convert standardized schema to model-native columns.

    Contract:
      - Same row count and index order as std_df
      - Include every column listed in manifest format.output_columns
    """
    out = pd.DataFrame(index=std_df.index)
    out["feature"] = pd.to_numeric(std_df["edit_len"], errors="raise")
    out["Efficiency"] = pd.to_numeric(std_df["editing_efficiency"], errors="raise")
    return out

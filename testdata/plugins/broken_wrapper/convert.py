from __future__ import annotations

import pandas as pd


def convert(std_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=std_df.index)
    out["feature"] = pd.to_numeric(std_df["edit_len"], errors="raise")
    out["Efficiency"] = pd.to_numeric(std_df["editing_efficiency"], errors="raise")
    return out

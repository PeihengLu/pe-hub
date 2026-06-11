"""Tests for /api/statistics aggregation over standardized entry rows."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _filter_dataframe_entries(
    df: pd.DataFrame,
    *,
    edit_lengths: list[int] | None = None,
) -> pd.DataFrame:
    """Mirror of DatasheetRepository._filter_dataframe_entries (edit-length path only)."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if edit_lengths is not None:
        length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
        if length_col not in df.columns:
            return df.iloc[0:0]
        mask &= pd.to_numeric(df[length_col], errors="coerce").isin(edit_lengths)

    return df.loc[mask]


def _extract_edit_length_series(df: pd.DataFrame) -> pd.Series:
    """Mirror of DatasheetRepository._extract_edit_length_series."""
    length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
    if length_col not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    return pd.to_numeric(df[length_col], errors="coerce")


def _entry_df(edit_lens: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edit_len": edit_lens,
            "type_sub": [True] * len(edit_lens),
            "type_ins": [False] * len(edit_lens),
            "type_del": [False] * len(edit_lens),
        }
    )


def _collect_lengths_with_fixed_loop(datasheets: list[pd.DataFrame]) -> list[int]:
    edit_lengths = None
    collected: list[int] = []

    for data in datasheets:
        filtered = _filter_dataframe_entries(data, edit_lengths=edit_lengths)
        entry_edit_lengths = _extract_edit_length_series(filtered)
        for idx in filtered.index:
            collected.append(int(entry_edit_lengths.loc[idx]))

    return collected


def _collect_lengths_with_buggy_loop(datasheets: list[pd.DataFrame]) -> list[int]:
    edit_lengths = None
    collected: list[int] = []

    for data in datasheets:
        filtered = _filter_dataframe_entries(data, edit_lengths=edit_lengths)
        edit_lengths = _extract_edit_length_series(filtered)
        for idx in filtered.index:
            collected.append(int(edit_lengths.loc[idx]))

    return collected


def test_statistics_loop_keeps_all_edit_lengths_across_datasheets():
    """Regression: do not reuse extracted edit-length series as the entry filter."""
    datasheets = [_entry_df([1]), _entry_df([1, 2, 3])]
    assert sorted(_collect_lengths_with_fixed_loop(datasheets)) == [1, 1, 2, 3]
    assert sorted(_collect_lengths_with_buggy_loop(datasheets)) == [1, 1]

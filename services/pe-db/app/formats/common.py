"""Shared helpers for standardized → model-format converters."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

STANDARDIZED_REQUIRED_COLUMNS = {
    "wt_sequence",
    "mut_sequence",
    "edit_len",
    "type_sub",
    "type_ins",
    "type_del",
    "protospacer_location_l",
    "protospacer_location_r",
    "pbs_location_l",
    "pbs_location_r",
    "rtt_location_l",
    "rtt_location_r",
    "lha_location_r",
}


def has_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    return set(required_columns).issubset(df.columns)


def is_standardized_dataframe(df: pd.DataFrame) -> bool:
    return has_columns(df, STANDARDIZED_REQUIRED_COLUMNS)


def _col_as_series(df: pd.DataFrame, colname: str, default: Any = 0) -> pd.Series:
    if colname in df.columns and isinstance(df[colname], pd.Series):
        return df[colname]
    return pd.Series(default, index=df.index)


def _edit_length_series(df: pd.DataFrame) -> pd.Series:
    """Read the canonical ``edit_len`` column."""
    return _col_as_series(df, "edit_len", 0)


def _safe_int_series(series: pd.Series, default: int = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    n_coerced = int(numeric.isna().sum())
    if n_coerced:
        # Falling back to 0 on a coordinate column silently relocates PBS/RT/
        # protospacer windows to the start of the sequence, which produces
        # plausible-looking but wrong model inputs. Warn loudly instead.
        logger.warning(
            "Column %r had %s non-numeric value(s) replaced with %s; "
            "downstream sequence slices for those rows are unreliable.",
            series.name,
            n_coerced,
            default,
        )
    return pd.Series(numeric, index=series.index).fillna(default).astype(int)


def _safe_float_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric, index=series.index).fillna(default).astype(float)


def _label_series(series: pd.Series) -> pd.Series:
    """Coerce a supervised-target column to float, keeping missing values as NaN.

    Deliberately does not fill like :func:`_safe_float_series`: substituting 0.0
    for an unmeasured efficiency is indistinguishable from a real measurement of
    zero, so model wrappers could not reject it. Standardization already drops
    unmeasured rows, so a NaN here means the frame did not come through that
    path and the wrapper should refuse to train on it.
    """
    return pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index).astype(float)


def _report_progress_milestone(
    progress_callback: Optional[ProgressCallback],
    *,
    phase: str,
    done: int,
    total: int,
    last_milestone: list[int],
) -> None:
    if progress_callback is None or total <= 0:
        return
    pct = int(100 * done / total)
    milestone = 100 if done >= total else (pct // 10) * 10
    if done >= total or milestone > last_milestone[0]:
        last_milestone[0] = milestone
        progress_callback(f"{phase}: {done}/{total} ({pct}%)")


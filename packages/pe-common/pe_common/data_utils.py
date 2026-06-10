"""Data splitting utilities shared across PE models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _stable_group_sort_key(value: Any) -> str:
    """Build a stable sort key across mixed Python scalar types."""
    return f"{type(value).__name__}:{value}"


def build_test_mask_from_group_id(
    group_series: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.Series:
    """
    Deterministically assign test groups from a group-id series.

    The selection depends only on:
      1) the set of non-null group IDs,
      2) test_size, and
      3) random_state.

    This means the same group-id set yields the same test groups even if row
    order differs between dataframes.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    groups = pd.Series(group_series, copy=False)
    unique_groups = groups.dropna().unique().tolist()
    if not unique_groups:
        return pd.Series(False, index=groups.index, dtype=bool)

    # Sort before shuffling so sampling is stable for the same set of groups.
    ordered_groups = sorted(unique_groups, key=_stable_group_sort_key)
    rng = np.random.default_rng(random_state)
    rng.shuffle(ordered_groups)

    n_test_groups = max(1, int(np.ceil(len(ordered_groups) * test_size)))
    test_groups = set(ordered_groups[:n_test_groups])
    return groups.isin(test_groups)


def extract_protospacer_series(
    df: pd.DataFrame,
    *,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.Series:
    """Extract protospacer sequences from standardized rows."""
    if wt_col not in df.columns:
        raise ValueError(f"Missing {wt_col!r} for protospacer extraction")
    wt = df[wt_col].astype(str)
    left = pd.to_numeric(
        df.get(protospacer_l_col, pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    right = pd.to_numeric(
        df.get(protospacer_r_col, pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)

    def _slice_protospacer(seq: str, l: int, r: int) -> str:
        if not isinstance(seq, str) or l < 0 or r <= l or r > len(seq):
            return ""
        return seq[l:r].upper()

    return pd.Series(
        (_slice_protospacer(seq, int(l), int(r)) for seq, l, r in zip(wt, left, right)),
        index=df.index,
        dtype="string",
    )


def target_location_group_series(
    df: pd.DataFrame,
    *,
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.Series:
    """Assign integer group ids so rows sharing a protospacer share a group."""
    protospacers = extract_protospacer_series(
        df,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    codes, _ = pd.factorize(protospacers, sort=False)
    return pd.Series(codes, index=df.index, dtype=int)


def reassign_group_ids_by_target_location(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    wt_col: str = "wt_sequence",
    protospacer_l_col: str = "protospacer_location_l",
    protospacer_r_col: str = "protospacer_location_r",
) -> pd.DataFrame:
    """Reassign ``group_col`` from protospacer so merged datasheets share groups by target."""
    output = df.copy()
    output[group_col] = target_location_group_series(
        output,
        wt_col=wt_col,
        protospacer_l_col=protospacer_l_col,
        protospacer_r_col=protospacer_r_col,
    )
    return output

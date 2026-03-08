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

"""Tests for universal target-locus identifiers in pe_common.data_utils."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pe-common"))

from pe_common.data_utils import (
    TARGET_UID_COLUMN,
    add_target_uid,
    compute_target_uid,
    target_uid_series,
)


def _frame(protospacers: list[str], *, pad_left: int = 4, pad_right: int = 10) -> pd.DataFrame:
    wt = [f"{'N' * pad_left}{ps}{'N' * pad_right}" for ps in protospacers]
    return pd.DataFrame(
        {
            "wt_sequence": wt,
            "protospacer_location_l": [pad_left] * len(protospacers),
            "protospacer_location_r": [pad_left + len(ps) for ps in protospacers],
        }
    )


def test_compute_target_uid_is_deterministic_and_protospacer_based():
    uid_a = compute_target_uid("ACGT" * 5)
    uid_b = compute_target_uid("acgt" * 5)  # case-insensitive
    assert uid_a == uid_b
    assert uid_a.startswith("ps:")


def test_compute_target_uid_falls_back_to_wt_sequence():
    uid = compute_target_uid("", wt_sequence="ACGTACGT")
    assert uid.startswith("wt:")
    assert compute_target_uid(None, None) == ""


def test_same_protospacer_yields_same_uid_across_datasets():
    shared = "A" * 20
    # Two "datasets" with different padding/flanks but the same protospacer.
    df1 = _frame([shared], pad_left=4, pad_right=10)
    df2 = _frame([shared], pad_left=7, pad_right=3)
    uid1 = target_uid_series(df1).iloc[0]
    uid2 = target_uid_series(df2).iloc[0]
    assert uid1 == uid2


def test_distinct_protospacers_yield_distinct_uids():
    df = _frame(["A" * 20, "C" * 20, "G" * 20])
    uids = target_uid_series(df).tolist()
    assert len(set(uids)) == 3


def test_add_target_uid_adds_column_without_mutating_input():
    df = _frame(["A" * 20, "A" * 20, "T" * 20])
    out = add_target_uid(df)
    assert TARGET_UID_COLUMN not in df.columns
    assert TARGET_UID_COLUMN in out.columns
    # Rows sharing a protospacer share the universal ID.
    assert out[TARGET_UID_COLUMN].iloc[0] == out[TARGET_UID_COLUMN].iloc[1]
    assert out[TARGET_UID_COLUMN].iloc[0] != out[TARGET_UID_COLUMN].iloc[2]

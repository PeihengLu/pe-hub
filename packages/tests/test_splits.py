"""Tests for pe_common.splits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pe-common"))

from pe_common.splits import (
    SplitConfig,
    assign_splits,
    exclude_test_partition,
    resolve_train_val_from_splits,
    select_evaluation_partition,
    split_config_from_params,
    validate_split_config,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "original_fold": [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, pd.NA, pd.NA],
            "value": list(range(8)),
        }
    )


def test_holdout_2_requires_fractions_to_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_split_config(
            SplitConfig(strategy="holdout_2", train_pct=0.7, test_pct=0.2)
        )


def test_holdout_2_rejects_cv_folds():
    with pytest.raises(ValueError, match="cv_folds"):
        validate_split_config(
            SplitConfig(strategy="holdout_2", train_pct=0.8, test_pct=0.2, cv_folds=5)
        )


def test_holdout_3_rejects_use_original_fold():
    with pytest.raises(ValueError, match="incompatible"):
        validate_split_config(
            SplitConfig(
                strategy="holdout_3",
                train_pct=0.7,
                val_pct=0.15,
                test_pct=0.15,
                use_original_fold=True,
            )
        )


def test_cv_rejects_train_and_val_pct():
    with pytest.raises(ValueError, match="only test_pct"):
        validate_split_config(
            SplitConfig(strategy="cv", cv_folds=5, train_pct=0.8, test_pct=0.2)
        )


def test_holdout_2_group_assignment_is_deterministic():
    df = pd.DataFrame({"group_id": list(range(10)), "original_fold": pd.NA})
    config = split_config_from_params(strategy="holdout_2", train_pct=0.8, test_pct=0.2, random_state=7)
    out1, _ = assign_splits(df, config)
    out2, _ = assign_splits(df, config)
    assert out1["split"].tolist() == out2["split"].tolist()


def test_use_original_fold_assigns_test_fold():
    df = _sample_df()
    config = split_config_from_params(
        strategy="holdout_2",
        train_pct=0.5,
        test_pct=0.5,
        use_original_fold=True,
        random_state=1,
    )
    out, summary = assign_splits(df, config)
    test_rows = out[out["group_id"] == 2]
    train_rows = out[out["group_id"].isin([0, 1])]
    synthetic_rows = out[out["group_id"] == 3]

    assert test_rows["split"].eq("test").all()
    assert test_rows["split_source"].eq("original_fold").all()
    assert train_rows["split"].eq("train").all()
    assert train_rows["split_source"].eq("original_fold").all()
    assert synthetic_rows["split_source"].eq("group_id").all()
    assert summary["by_source"]["original_fold"] == 6
    assert summary["by_source"]["group_id"] == 2


def test_holdout_3_errors_when_original_fold_present():
    df = _sample_df()
    config = split_config_from_params(
        strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
        random_state=1,
    )
    with pytest.raises(ValueError, match="original_fold metadata is present"):
        assign_splits(df, config)


def test_cv_with_original_fold_maps_folds():
    df = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2],
            "original_fold": [0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
        }
    )
    config = split_config_from_params(strategy="cv", cv_folds=3, use_original_fold=True)
    out, _ = assign_splits(df, config)
    assert set(out.loc[out["group_id"] == 0, "split"]) == {"fold_0"}
    assert set(out.loc[out["group_id"] == 1, "split"]) == {"fold_1"}
    assert set(out.loc[out["group_id"] == 2, "split"]) == {"test"}


def test_composite_group_prefix_applied_on_merge():
    df = pd.DataFrame({"group_id": [0, 0, 1, 1], "original_fold": pd.NA})
    config = split_config_from_params(strategy="cv", cv_folds=2, random_state=3)
    out_a, _ = assign_splits(df, config, composite_group_prefix="sheet-a")
    out_b, _ = assign_splits(df, config, composite_group_prefix="sheet-b")
    assert out_a["split"].tolist() != out_b["split"].tolist()


def test_exclude_test_partition():
    df = pd.DataFrame({"split": ["train", "train", "test", "val"], "value": [1, 2, 3, 4]})
    out = exclude_test_partition(df)
    assert set(out["split"]) == {"train", "val"}


def test_resolve_train_val_holdout_3():
    df = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2],
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    )
    train, val = resolve_train_val_from_splits(df)
    assert len(train) == 2
    assert len(val) == 2


def test_select_evaluation_partition():
    df = pd.DataFrame({"split": ["train", "test", "test"], "value": [1, 2, 3]})
    test_df = select_evaluation_partition(df)
    assert len(test_df) == 2
    assert test_df["value"].tolist() == [2, 3]


def test_conflicting_original_fold_within_group_raises():
    df = pd.DataFrame({"group_id": [0, 0], "original_fold": [0.0, 1.0]})
    config = split_config_from_params(
        strategy="holdout_2",
        train_pct=0.5,
        test_pct=0.5,
        use_original_fold=True,
    )
    with pytest.raises(ValueError, match="Conflicting original_fold"):
        assign_splits(df, config)

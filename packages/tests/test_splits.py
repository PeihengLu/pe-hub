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
    protospacers = [
        "A" * 20,
        "A" * 20,
        "C" * 20,
        "C" * 20,
        "G" * 20,
        "G" * 20,
        "T" * 20,
        "T" * 19 + "A",
    ]
    wt_sequences = [f"{'N' * 4}{protospacer}{'N' * 10}" for protospacer in protospacers]
    return pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "original_fold": [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, pd.NA, pd.NA],
            "wt_sequence": wt_sequences,
            "protospacer_location_l": [4] * 8,
            "protospacer_location_r": [24] * 8,
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
        original_fold_test_value=-1.0,
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


def test_holdout_3_without_original_fold_splits_test_then_train_val():
    df = _sample_df()
    config = split_config_from_params(
        strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
        use_original_fold=False,
        random_state=1,
    )
    out, summary = assign_splits(df, config)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out["split_source"].eq("group_id").all()
    assert summary["by_source"]["group_id"] == len(df)
    train, val = resolve_train_val_from_splits(out)
    assert len(train) > 0
    assert len(val) > 0


def test_holdout_3_with_original_fold_uses_holdout_2_test_then_splits_train_val():
    df = _sample_df()
    config = split_config_from_params(
        strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
        use_original_fold=True,
        original_fold_test_value=-1.0,
        random_state=1,
    )
    out, summary = assign_splits(df, config)
    assert out.loc[out["group_id"] == 2, "split"].eq("test").all()
    assert out.loc[out["group_id"] == 2, "split_source"].eq("original_fold").all()
    assert set(out["split"]) == {"train", "val", "test"}
    assert summary["by_source"]["original_fold"] == 6
    train, val = resolve_train_val_from_splits(out)
    assert len(train) > 0
    assert len(val) > 0


def test_cv_with_original_fold_maps_folds():
    protospacers = ["A" * 20, "A" * 20, "C" * 20, "C" * 20, "G" * 20, "G" * 20]
    df = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2],
            "original_fold": [0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
            "wt_sequence": [f"{'N' * 4}{protospacer}{'N' * 10}" for protospacer in protospacers],
            "protospacer_location_l": [4] * 6,
            "protospacer_location_r": [24] * 6,
        }
    )
    config = split_config_from_params(strategy="cv", cv_folds=3, use_original_fold=True)
    out, _ = assign_splits(df, config)
    assert set(out.loc[out["group_id"] == 0, "split"]) == {"fold_0"}
    assert set(out.loc[out["group_id"] == 1, "split"]) == {"fold_1"}
    assert set(out.loc[out["group_id"] == 2, "split"]) == {"test"}


def test_composite_group_prefix_applied_on_merge():
    protospacers = ["A" * 20, "A" * 20, "C" * 20, "C" * 20]
    df = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1],
            "original_fold": pd.NA,
            "wt_sequence": [f"{'N' * 4}{protospacer}{'N' * 10}" for protospacer in protospacers],
            "protospacer_location_l": [4] * 4,
            "protospacer_location_r": [24] * 4,
        }
    )
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


def test_use_original_fold_ignores_group_id_conflicts():
    df = pd.DataFrame(
        {
            "group_id": [0, 0],
            "original_fold": [0.0, 1.0],
            "wt_sequence": ["A" * 30, "A" * 30],
            "protospacer_location_l": [4, 4],
            "protospacer_location_r": [24, 24],
        }
    )
    config = split_config_from_params(
        strategy="cv",
        cv_folds=2,
        use_original_fold=True,
    )
    out, _ = assign_splits(df, config)
    assert set(out.loc[out["original_fold"] == 0.0, "split"]) == {"fold_0"}
    assert set(out.loc[out["original_fold"] == 1.0, "split"]) == {"fold_1"}
    assert out["split_source"].eq("original_fold").all()


def test_custom_original_fold_test_value_for_pridict2_style_folds():
    df = pd.DataFrame(
        {
            "group_id": [0, 0, 1, 1, 2, 2],
            "original_fold": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "wt_sequence": [f"{'N' * 4}{'A' * 20}{'N' * 10}"] * 6,
            "protospacer_location_l": [4] * 6,
            "protospacer_location_r": [24] * 6,
        }
    )
    config = split_config_from_params(
        strategy="holdout_2",
        train_pct=0.5,
        test_pct=0.5,
        use_original_fold=True,
        original_fold_test_value=1.0,
    )
    out, _ = assign_splits(df, config)
    assert set(out.loc[out["group_id"] == 1, "split"]) == {"test"}
    assert set(out.loc[out["group_id"] == 0, "split"]) == {"train"}
    assert set(out.loc[out["group_id"] == 2, "split"]) == {"train"}


def test_merged_group_ids_reassigned_by_protospacer():
    from pe_common.data_utils import reassign_group_ids_by_target_location

    shared_protospacer = "A" * 20
    distinct_protospacer = "C" * 20
    df = pd.DataFrame(
        {
            "group_id": [0, 1, 2, 3],
            "wt_sequence": [
                f"{'N' * 4}{shared_protospacer}{'N' * 10}",
                f"{'N' * 4}{shared_protospacer}{'N' * 10}",
                f"{'N' * 4}{distinct_protospacer}{'N' * 10}",
                f"{'N' * 4}{distinct_protospacer}{'N' * 10}",
            ],
            "protospacer_location_l": [4, 4, 4, 4],
            "protospacer_location_r": [24, 24, 24, 24],
            "original_fold": pd.NA,
        }
    )
    reassigned = reassign_group_ids_by_target_location(df)
    assert reassigned["group_id"].tolist() == [0, 0, 1, 1]

    config = split_config_from_params(strategy="holdout_2", train_pct=0.5, test_pct=0.5, random_state=3)
    out, _ = assign_splits(reassigned, config)
    assert out.loc[out["group_id"] == 0, "split"].nunique() == 1
    assert out.loc[out["group_id"] == 1, "split"].nunique() == 1
    assert out["split"].nunique() == 2

"""Unit tests for datasheet-benchmark protocol helpers (no GPU / peen jobs)."""
from __future__ import annotations

import pandas as pd
import pytest

from protocol import (
    ProtocolError,
    assign_cv_folds,
    assign_holdout_3,
    choose_protocol,
    extract_eval_metrics,
    fold_labels,
    remap_cv_fold_to_nested_holdout,
    repeat_seeds,
    split_counts,
    summarize_repeats,
)


def _frame(n_groups: int, rows_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for group in range(n_groups):
        for extra in range(rows_per_group):
            rows.append(
                {
                    "target_uid": f"ps:{group:03d}",
                    "value": float(group * 10 + extra),
                }
            )
    return pd.DataFrame(rows)


def test_choose_protocol_auto_small():
    protocol, reason = choose_protocol(100, size_threshold=1000)
    assert protocol == "cv"
    assert "small" in reason


def test_choose_protocol_auto_large():
    protocol, reason = choose_protocol(80_000, size_threshold=50_000)
    assert protocol == "holdout_3"
    assert "large" in reason


def test_choose_protocol_override():
    protocol, reason = choose_protocol(10, size_threshold=5, override="holdout_3")
    assert protocol == "holdout_3"
    assert "override" in reason


def test_repeat_seeds_are_offset():
    assert repeat_seeds(42, 3) == [42, 43, 44]


def test_assign_holdout_3_prefers_target_uid_over_colliding_group_id():
    """Per-sheet group_id integers collide after pooling; target_uid is the locus key."""
    rows = []
    for sheet in (0, 1):
        for locus in range(6):
            rows.append(
                {
                    "group_id": locus,  # same integers on both sheets
                    "target_uid": f"sheet{sheet}:locus{locus}",
                    "value": float(sheet * 10 + locus),
                }
            )
    df = pd.DataFrame(rows)
    assigned = assign_holdout_3(df, random_state=0)
    by_uid = assigned.groupby("target_uid")["split"].nunique()
    assert (by_uid == 1).all()
    by_group = assigned.groupby("group_id")["split"].nunique()
    assert (by_group > 1).any()


def test_assign_holdout_3_partitions_and_is_deterministic():
    df = _frame(20)
    first = assign_holdout_3(df, random_state=7)
    second = assign_holdout_3(df, random_state=7)
    third = assign_holdout_3(df, random_state=8)
    counts = split_counts(first)
    assert set(counts) == {"train", "val", "test"}
    assert counts["train"] > counts["val"]
    assert first["split"].tolist() == second["split"].tolist()
    assert first["split"].tolist() != third["split"].tolist()


def test_assign_holdout_3_rejects_too_few_groups():
    with pytest.raises(ProtocolError, match="at least 3"):
        assign_holdout_3(_frame(2), random_state=0)


def test_cv_nested_holdout_keeps_outer_fold_as_test():
    df = _frame(15)
    cv = assign_cv_folds(df, cv_folds=5, random_state=42)
    labels = fold_labels(cv)
    assert labels == ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
    nested = remap_cv_fold_to_nested_holdout(cv, "fold_2", inner_random_state=99)
    counts = split_counts(nested)
    assert set(counts) == {"train", "val", "test"}
    outer = cv.loc[cv["split"].astype(str) == "fold_2", "target_uid"]
    nested_test = nested.loc[nested["split"].astype(str) == "test", "target_uid"]
    assert set(outer) == set(nested_test)
    # Inner train/val must not include the outer test loci.
    inner_uids = set(
        nested.loc[nested["split"].astype(str) != "test", "target_uid"]
    )
    assert inner_uids.isdisjoint(set(outer))


def test_cv_rejects_too_few_groups():
    with pytest.raises(ProtocolError, match="at least 5"):
        assign_cv_folds(_frame(3), cv_folds=5, random_state=1)


def test_extract_eval_metrics_aliases():
    payload = {"metrics": {"averageedited_spearman": 0.4, "pearson": 0.3, "mse": 1.2}}
    metrics = extract_eval_metrics(payload)
    assert metrics["spearman"] == pytest.approx(0.4)
    assert metrics["pearson"] == pytest.approx(0.3)
    assert metrics["mse"] == pytest.approx(1.2)


def test_summarize_repeats_mean_std():
    rows = [
        {"status": "ok", "test_spearman": 0.5, "test_pearson": 0.4, "test_mse": 1.0},
        {"status": "ok", "test_spearman": 0.7, "test_pearson": 0.6, "test_mse": 3.0},
        {"status": "error", "test_spearman": 0.9},
    ]
    summary = summarize_repeats(rows)
    assert summary["n_ok"] == 2
    assert summary["test_spearman_mean"] == pytest.approx(0.6)
    assert summary["test_spearman_std"] == pytest.approx(0.141421356, rel=1e-5)
    assert summary["test_mse_mean"] == pytest.approx(2.0)


def test_merge_repeat_rows_keeps_all_seeds():
    from run_benchmark import merge_repeat_rows

    merged = merge_repeat_rows(
        [{"repeat_id": "seed_43", "status": "ok", "seed": 43}],
        [
            {"repeat_id": "seed_42", "status": "ok", "seed": 42},
            {"repeat_id": "seed_43", "status": "tuned", "seed": 43},
        ],
    )
    assert [row["repeat_id"] for row in merged] == ["seed_42", "seed_43"]
    assert merged[1]["status"] == "tuned"

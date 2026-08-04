"""Detect train/test data leakage before running an evaluation.

A model's evaluation is only meaningful when its test data was not seen during
training. This module compares the universal target-locus IDs (``target_uid``)
of an evaluation's test partition against the loci a weight set was trained on
(recorded at training time by :mod:`app.models.weights_registry`).

When leakage is detected -- or cannot be ruled out (e.g. the dataset provided
no original test split and the weight set has no recorded provenance) -- a
structured, machine-parseable result is emitted instead of misleading metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, target_uid_series

from ..models import weights_registry

LEAK_ERROR_TYPE = "data_leak"

# Leak reasons (stable identifiers for downstream parsing):
REASON_TRAIN_TEST_OVERLAP = "train_test_overlap"
REASON_NO_ORIGINAL_TEST_SPLIT = "no_original_test_split"
REASON_UNVERIFIABLE_PROVENANCE = "unverifiable_provenance"

_MAX_EXAMPLE_UIDS = 20


@dataclass(frozen=True)
class LeakAssessment:
    """Outcome of a leakage check for one evaluation run."""

    is_leak: bool
    reason: str
    detail: Dict[str, Any]


def _clean_uid_set(values) -> set[str]:
    return {str(value) for value in values if value and str(value) not in ("", "nan", "<NA>")}


def _test_target_uids(test_df: pd.DataFrame) -> Optional[set[str]]:
    """Resolve universal target-locus IDs for the test partition."""
    if TARGET_UID_COLUMN in test_df.columns:
        return _clean_uid_set(test_df[TARGET_UID_COLUMN].dropna().tolist())
    # Inline records may lack target_uid; recompute if standardized columns exist.
    if "wt_sequence" in test_df.columns:
        try:
            uids = target_uid_series(test_df)
        except Exception:
            return None
        return _clean_uid_set(uids.dropna().tolist())
    return None


def _test_split_source_counts(test_df: pd.DataFrame) -> Dict[str, int]:
    if "split_source" not in test_df.columns or test_df.empty:
        return {}
    counts = (
        test_df["split_source"].astype("string").fillna("none").value_counts().to_dict()
    )
    return {str(key): int(value) for key, value in counts.items()}


def assess_leakage(
    *,
    test_df: pd.DataFrame,
    split: Any,
    model: str,
    weights_id: str,
) -> Optional[LeakAssessment]:
    """Return a :class:`LeakAssessment` when leakage is present or unverifiable.

    Returns ``None`` when the evaluation is provably (or acceptably) leak-free.
    Never raises; on any internal error it degrades to ``None`` so that a
    provenance-check bug cannot take down evaluation.
    """
    try:
        return _assess_leakage(
            test_df=test_df, split=split, model=model, weights_id=weights_id
        )
    except Exception:
        return None


def _assess_leakage(
    *,
    test_df: pd.DataFrame,
    split: Any,
    model: str,
    weights_id: str,
) -> Optional[LeakAssessment]:
    training_loci = weights_registry.load_training_loci(model, weights_id)
    test_uids = _test_target_uids(test_df)
    source_counts = _test_split_source_counts(test_df)

    n_total = int(len(test_df))
    n_author = int(source_counts.get("original_fold", 0))
    n_synthetic = n_total - n_author
    test_is_author_holdout = n_author > 0 and n_synthetic == 0
    use_original_fold = bool(getattr(split, "use_original_fold", False))

    base_detail: Dict[str, Any] = {
        "weights_id": weights_id,
        "training_provenance_available": training_loci is not None,
        "n_test_rows": n_total,
        "n_test_loci": len(test_uids) if test_uids is not None else None,
        "n_training_loci": len(training_loci) if training_loci is not None else None,
        "test_split_source": source_counts,
        "use_original_fold": use_original_fold,
    }

    # Case 1: training provenance recorded -> authoritative overlap check.
    if training_loci is not None:
        if test_uids is None:
            # Cannot compute test loci (e.g. inline records without target_uid);
            # avoid false positives and let the evaluation proceed.
            return None
        overlap = sorted(test_uids & training_loci)
        detail = {
            **base_detail,
            "n_overlap_loci": len(overlap),
            "overlap_fraction": (len(overlap) / len(test_uids)) if test_uids else 0.0,
            "example_overlap_target_uids": overlap[:_MAX_EXAMPLE_UIDS],
        }
        if overlap:
            detail["message"] = (
                f"{len(overlap)} of {len(test_uids)} evaluation target loci were "
                "present in this model's training data (train/test overlap)."
            )
            return LeakAssessment(True, REASON_TRAIN_TEST_OVERLAP, detail)
        return None

    # Case 2: no recorded provenance (e.g. vendor pretrained weights).
    if test_is_author_holdout:
        # The test rows come from an author-designated held-out split; trust it.
        return None

    if use_original_fold:
        reason = REASON_NO_ORIGINAL_TEST_SPLIT
        message = (
            "Data leak unavoidable: an original (author-provided) test split was "
            "requested but is not defined for this benchmark, so the test set was "
            "synthesized, and the weight set has no recorded training provenance "
            "to verify separation."
        )
    else:
        reason = REASON_UNVERIFIABLE_PROVENANCE
        message = (
            "Cannot verify train/test separation: the test split is synthetic and "
            "the weight set has no recorded training provenance."
        )
    detail = {**base_detail, "message": message}
    return LeakAssessment(True, reason, detail)


def leak_error_payload(
    assessment: LeakAssessment,
    *,
    model: str,
    benchmark_name: str,
    weights: str,
    device_id: str,
    n_samples: int,
) -> Dict[str, Any]:
    """Build the parseable error result emitted when evaluation is aborted."""
    return {
        "model": model,
        "benchmark_name": benchmark_name,
        "weights": weights,
        "device": device_id,
        "status": "error",
        "error_type": LEAK_ERROR_TYPE,
        "leak_reason": assessment.reason,
        "leak": assessment.detail,
        "n_samples": int(n_samples),
        "metrics": None,
    }

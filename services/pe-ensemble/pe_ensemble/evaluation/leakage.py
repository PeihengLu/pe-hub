"""Detect train/test data leakage before running an evaluation.

A model's evaluation is only meaningful when its test data was not seen during
training. This module compares the universal target-locus IDs (``target_uid``)
of an evaluation's test partition against the loci a weight set was trained on
(recorded at training time by :mod:`pe_ensemble.models.weights_registry`).

When leakage is detected -- or cannot be ruled out (e.g. the dataset provided
no original test split and the weight set has no recorded provenance) -- a
structured, machine-parseable result is emitted instead of misleading metrics.

Sheets with **no author test split for this weight** cannot be turned into a
valid test by taking a random holdout or by dropping leftover loci after a
UID match (``no_original_test_split``). That includes:

- PRIDICT library1, OptiPrime lib-mmr / lib-cv (no official test split)
- in-domain eval of a weight whose provenance sets
  ``has_original_test_split`` false, even when the *sheet* has another paper's
  ``original_fold`` (OptiPrime's pooled protospacer CV is not Yu's ClinVar
  holdout or Mathis's library-diverse folds). Leftover-excluding those rows
  would score a non-random remainder, not this model's test.

For an **author-defined** holdout that this weight actually used, and that
still overlaps recorded training loci, the default is to **exclude**
overlapping target loci and continue when at least one locus remains. Full
overlap still aborts unless ``allow_data_leak`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, target_uid_series

from ..models import weights_registry

LEAK_ERROR_TYPE = "data_leak"

# Leak reasons (stable identifiers for downstream parsing):
REASON_TRAIN_TEST_OVERLAP = "train_test_overlap"
REASON_NO_ORIGINAL_TEST_SPLIT = "no_original_test_split"
REASON_UNVERIFIABLE_PROVENANCE = "unverifiable_provenance"

_MAX_EXAMPLE_UIDS = 20
_DATASET_FRAME_COLUMNS = ("dataset", "dataset_name")


def normalize_dataset_name(value: Any) -> str:
    """Lowercase hyphenated dataset key (``library_diverse`` → ``library-diverse``)."""
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "<na>", "none"}:
        return ""
    return text.replace("_", "-")


def dataset_names_from_value(value: Any) -> set[str]:
    """Normalize a filter value (scalar, list, or None) to dataset name keys."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    names = {normalize_dataset_name(item) for item in items}
    return {name for name in names if name}


def dataset_names_from_training(training: Optional[Dict[str, Any]]) -> set[str]:
    """Dataset names recorded on a weight set's ``training`` block."""
    if not isinstance(training, dict):
        return set()
    names = dataset_names_from_value((training.get("filters") or {}).get("dataset"))
    provenance = training.get("data_provenance")
    if isinstance(provenance, dict):
        for entry in provenance.get("vendor_training_lineage") or []:
            if isinstance(entry, dict):
                names |= dataset_names_from_value(entry.get("dataset"))
    return names


def weight_respects_author_holdout(training: Optional[Dict[str, Any]]) -> bool:
    """Whether leftover-exclude is allowed on another study's ``original_fold``.

    Provenance ``has_original_test_split: false`` means this weight did not
    use that holdout (e.g. OptiPrime ``base``). A missing key keeps the
    previous leftover-exclude behavior for DeepPrime / PRIDICT2 / OPED.
    """
    if not isinstance(training, dict):
        return True
    provenance = training.get("data_provenance")
    if not isinstance(provenance, dict):
        return True
    if "has_original_test_split" not in provenance:
        return True
    return bool(provenance["has_original_test_split"])


def dataset_names_from_frame(frame: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    for column in _DATASET_FRAME_COLUMNS:
        if column in frame.columns:
            names |= dataset_names_from_value(frame[column].dropna().tolist())
    return names


@dataclass(frozen=True)
class LeakAssessment:
    """Outcome of a leakage check for one evaluation run."""

    is_leak: bool
    reason: str
    detail: Dict[str, Any]


@dataclass(frozen=True)
class LeakExclusion:
    """Result of dropping training-overlapping loci from a test partition."""

    filtered_df: pd.DataFrame
    overlap_uids: tuple[str, ...]
    n_rows_before: int
    n_rows_after: int
    n_loci_before: int
    n_loci_after: int

    @property
    def n_overlap_loci(self) -> int:
        return len(self.overlap_uids)

    @property
    def is_empty(self) -> bool:
        return self.filtered_df.empty

    def warning_payload(self, *, reason: str = REASON_TRAIN_TEST_OVERLAP) -> Dict[str, Any]:
        return {
            "reason": reason,
            "action": "excluded_overlap_loci",
            "n_overlap_loci": self.n_overlap_loci,
            "n_test_loci_before": self.n_loci_before,
            "n_test_loci_after": self.n_loci_after,
            "n_test_rows_before": self.n_rows_before,
            "n_test_rows_after": self.n_rows_after,
            "example_overlap_target_uids": list(self.overlap_uids[:_MAX_EXAMPLE_UIDS]),
            "message": (
                f"Excluded {self.n_overlap_loci} overlapping target loci "
                f"({self.n_rows_before - self.n_rows_after} rows) from the test "
                f"partition; evaluating {self.n_loci_after} remaining loci "
                f"({self.n_rows_after} rows)."
            ),
        }


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


def restrict_to_author_holdout_rows(test_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep author ``original_fold`` test rows when they are mixed with unlabeled rows.

    DeepPE HEK pools HT/type/position (``original_fold=-1``) with endo (no author
    fold). Unlabeled endo groups can land in the synthetic test slice; those rows
    are not an author holdout and would make an otherwise valid test look
    synthetic to leak checks.
    """
    if test_df.empty or "split_source" not in test_df.columns:
        return test_df, 0
    author_mask = test_df["split_source"].astype("string").eq("original_fold")
    if not bool(author_mask.any()) or bool(author_mask.all()):
        return test_df, 0
    n_dropped = int((~author_mask).sum())
    return test_df.loc[author_mask].copy(), n_dropped


def _row_target_uids(test_df: pd.DataFrame) -> Optional[pd.Series]:
    if TARGET_UID_COLUMN in test_df.columns:
        return test_df[TARGET_UID_COLUMN].astype("string")
    if "wt_sequence" in test_df.columns:
        try:
            return target_uid_series(test_df).astype("string")
        except Exception:
            return None
    return None


def exclude_overlapping_loci(
    test_df: pd.DataFrame,
    training_loci: Set[str],
) -> Optional[LeakExclusion]:
    """Drop rows whose ``target_uid`` is in ``training_loci``.

    Returns ``None`` when target UIDs cannot be resolved (caller should not
    invent a filter). Returns an exclusion with an empty frame when every
    test locus overlaps training.
    """
    if test_df.empty or not training_loci:
        return LeakExclusion(
            filtered_df=test_df.copy(),
            overlap_uids=tuple(),
            n_rows_before=int(len(test_df)),
            n_rows_after=int(len(test_df)),
            n_loci_before=0,
            n_loci_after=0,
        )

    uid_series = _row_target_uids(test_df)
    if uid_series is None:
        return None

    cleaned = uid_series.fillna("").astype(str)
    test_uids = _clean_uid_set(cleaned.tolist())
    overlap = sorted(test_uids & set(training_loci))
    if not overlap:
        return LeakExclusion(
            filtered_df=test_df.copy(),
            overlap_uids=tuple(),
            n_rows_before=int(len(test_df)),
            n_rows_after=int(len(test_df)),
            n_loci_before=len(test_uids),
            n_loci_after=len(test_uids),
        )

    overlap_set = set(overlap)
    keep_mask = ~cleaned.isin(overlap_set)
    # Also drop rows with missing UIDs when any overlap exists? Keep them —
    # missing UID cannot be proven to overlap.
    filtered = test_df.loc[keep_mask].reset_index(drop=True)
    remaining_uids = _clean_uid_set(cleaned.loc[keep_mask].tolist())
    return LeakExclusion(
        filtered_df=filtered,
        overlap_uids=tuple(overlap),
        n_rows_before=int(len(test_df)),
        n_rows_after=int(len(filtered)),
        n_loci_before=len(test_uids),
        n_loci_after=len(remaining_uids),
    )


def _member_identity(member: Any) -> Tuple[str, str]:
    if isinstance(member, dict):
        model_name = str(member.get("model_name") or member.get("model") or "").strip().lower()
        weights = str(member.get("weights") or "").strip()
    else:
        model_name = str(getattr(member, "model_name", "") or "").strip().lower()
        weights = str(getattr(member, "weights", "") or "").strip()
    return model_name, weights


def collect_ensemble_training_loci(
    members: Sequence[Any],
) -> Tuple[Optional[Set[str]], Dict[str, Any]]:
    """Union ``train_target_loci`` from each ensemble member's provenance.

    Returns ``(training_loci, detail)``. ``training_loci`` is the union when
    every member has a recorded sidecar; ``None`` when any member lacks
    provenance (ensemble training exposure cannot be fully known).
    """
    member_rows: list[Dict[str, Any]] = []
    union: Set[str] = set()
    missing: list[Dict[str, str]] = []

    for member in members:
        model_name, weights_id = _member_identity(member)
        loci = weights_registry.load_training_loci(model_name, weights_id)
        row: Dict[str, Any] = {
            "model_name": model_name,
            "weights": weights_id,
            "training_provenance_available": loci is not None,
            "n_target_loci": len(loci) if loci is not None else None,
        }
        member_rows.append(row)
        if loci is None:
            missing.append({"model_name": model_name, "weights": weights_id})
        else:
            union |= set(loci)

    complete = len(missing) == 0 and len(member_rows) > 0
    detail: Dict[str, Any] = {
        "members": member_rows,
        "all_members_have_provenance": complete,
        "n_members_missing_provenance": len(missing),
        "missing_provenance_members": missing,
        "n_target_loci": len(union) if complete else None,
        "loci_fingerprint": (
            weights_registry.loci_fingerprint(union) if complete and union else None
        ),
    }
    return (union if complete else None), detail


def collect_ensemble_training_datasets(
    members: Sequence[Any],
) -> set[str]:
    """Union of dataset names recorded on each ensemble member's training block."""
    names: set[str] = set()
    for member in members:
        model_name, weights_id = _member_identity(member)
        names |= dataset_names_from_training(
            weights_registry.load_training_metadata(model_name, weights_id)
        )
    return names


def assess_leakage(
    *,
    test_df: pd.DataFrame,
    split: Any,
    model: str,
    weights_id: str,
    eval_datasets: Any = None,
) -> Optional[LeakAssessment]:
    """Return a :class:`LeakAssessment` when leakage is present or unverifiable.

    Returns ``None`` when the evaluation is provably (or acceptably) leak-free.
    Never raises; on any internal error it degrades to ``None`` so that a
    provenance-check bug cannot take down evaluation.
    """
    try:
        training = weights_registry.load_training_metadata(model, weights_id)
        training_loci = weights_registry.load_training_loci(model, weights_id)
        training_datasets = dataset_names_from_training(training)
        return _assess_leakage(
            test_df=test_df,
            split=split,
            training_loci=training_loci,
            training_datasets=training_datasets,
            eval_datasets=eval_datasets,
            respects_author_holdout=weight_respects_author_holdout(training),
            base_detail={
                "weights_id": weights_id,
                "training_provenance_available": training_loci is not None,
            },
        )
    except Exception:
        return None


def assess_ensemble_leakage(
    *,
    test_df: pd.DataFrame,
    split: Any,
    members: Sequence[Any],
    eval_datasets: Any = None,
) -> Optional[LeakAssessment]:
    """Leak check for an ensemble using the union of member training loci."""
    try:
        training_loci, loci_detail = collect_ensemble_training_loci(members)
        training_datasets = collect_ensemble_training_datasets(members)
        respects_author_holdout = True
        for member in members:
            model_name, weights_id = _member_identity(member)
            respects_author_holdout = respects_author_holdout and weight_respects_author_holdout(
                weights_registry.load_training_metadata(model_name, weights_id)
            )
        return _assess_leakage(
            test_df=test_df,
            split=split,
            training_loci=training_loci,
            training_datasets=training_datasets,
            eval_datasets=eval_datasets,
            respects_author_holdout=respects_author_holdout,
            base_detail={
                "ensemble_training_loci": loci_detail,
                "training_provenance_available": training_loci is not None,
            },
            overlap_message_subject="this ensemble's member training data",
            unverifiable_subject="one or more ensemble members have",
        )
    except Exception:
        return None


def _assess_leakage(
    *,
    test_df: pd.DataFrame,
    split: Any,
    training_loci: Optional[Set[str]],
    base_detail: Dict[str, Any],
    training_datasets: Optional[Set[str]] = None,
    eval_datasets: Any = None,
    respects_author_holdout: bool = True,
    overlap_message_subject: str = "this model's training data",
    unverifiable_subject: str = "the weight set has",
) -> Optional[LeakAssessment]:
    test_uids = _test_target_uids(test_df)
    source_counts = _test_split_source_counts(test_df)

    n_total = int(len(test_df))
    n_author = int(source_counts.get("original_fold", 0))
    n_synthetic = n_total - n_author
    test_is_author_holdout = n_author > 0 and n_synthetic == 0
    use_original_fold = bool(getattr(split, "use_original_fold", False))
    eval_names = dataset_names_from_value(eval_datasets) | dataset_names_from_frame(test_df)
    train_names = set(training_datasets or ())
    in_domain = sorted(eval_names & train_names)

    detail_base: Dict[str, Any] = {
        **base_detail,
        "n_test_rows": n_total,
        "n_test_loci": len(test_uids) if test_uids is not None else None,
        "n_training_loci": len(training_loci) if training_loci is not None else None,
        "test_split_source": source_counts,
        "use_original_fold": use_original_fold,
        "eval_datasets": sorted(eval_names),
        "training_datasets": sorted(train_names),
        "in_domain_datasets": in_domain,
        "test_is_author_holdout": test_is_author_holdout,
        "weight_respects_author_holdout": bool(respects_author_holdout),
    }

    # In-domain eval of a training sheet when leftover-exclude would not be a
    # valid test: synthetic split, or another paper's original_fold that this
    # weight did not use (OptiPrime ``has_original_test_split: false``).
    leftover_exclude_invalid = in_domain and (
        not test_is_author_holdout or not respects_author_holdout
    )
    if leftover_exclude_invalid:
        joined = ", ".join(in_domain)
        if test_is_author_holdout:
            message = (
                f"Data leak unavoidable: evaluating {joined}, and "
                f"{overlap_message_subject} includes that dataset, but this "
                "weight set did not use the sheet's author holdout. Leftover "
                "loci after UID exclusion would not be a valid test."
            )
        else:
            message = (
                f"Data leak unavoidable: evaluating {joined} with a synthetic "
                f"test split, and {overlap_message_subject} includes that "
                "dataset. This evaluation is not using an author-defined "
                "holdout; leftover loci after UID exclusion would not be a "
                "valid test."
            )
        detail = {**detail_base, "message": message}
        return LeakAssessment(True, REASON_NO_ORIGINAL_TEST_SPLIT, detail)

    # Case 1: training provenance recorded -> authoritative overlap check.
    if training_loci is not None:
        if test_uids is None:
            # Cannot compute test loci (e.g. inline records without target_uid);
            # avoid false positives and let the evaluation proceed.
            return None
        overlap = sorted(test_uids & training_loci)
        detail = {
            **detail_base,
            "n_overlap_loci": len(overlap),
            "overlap_fraction": (len(overlap) / len(test_uids)) if test_uids else 0.0,
            "example_overlap_target_uids": overlap[:_MAX_EXAMPLE_UIDS],
        }
        if overlap:
            detail["message"] = (
                f"{len(overlap)} of {len(test_uids)} evaluation target loci were "
                f"present in {overlap_message_subject} (train/test overlap)."
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
            "synthesized, and "
            f"{unverifiable_subject} no recorded training provenance "
            "to verify separation."
        )
    else:
        reason = REASON_UNVERIFIABLE_PROVENANCE
        message = (
            "Cannot verify train/test separation: the test split is synthetic and "
            f"{unverifiable_subject} no recorded training provenance."
        )
    detail = {**detail_base, "message": message}
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


def ensemble_leak_error_payload(
    assessment: LeakAssessment,
    *,
    ensemble_name: str,
    device_id: str,
    n_samples: int,
    members: Iterable[Any],
) -> Dict[str, Any]:
    """Build the parseable error result emitted when ensemble evaluation aborts."""
    member_list = []
    for member in members:
        model_name, weights = _member_identity(member)
        member_list.append({"model_name": model_name, "weights": weights})
    return {
        "ensemble_name": ensemble_name,
        "device": device_id,
        "status": "error",
        "error_type": LEAK_ERROR_TYPE,
        "leak_reason": assessment.reason,
        "leak": assessment.detail,
        "n_samples": int(n_samples),
        "metrics": None,
        "member_metrics": [],
        "members": member_list,
    }

"""Tests for train/test data-leak detection during evaluation."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.evaluation import leakage
from app.evaluation.leakage import (
    REASON_NO_ORIGINAL_TEST_SPLIT,
    REASON_TRAIN_TEST_OVERLAP,
    REASON_UNVERIFIABLE_PROVENANCE,
    assess_leakage,
)
from app.evaluation.runner import execute_evaluation
from app.evaluation.schemas import EvaluationRequest
from app.training.data import ModelFormatFetchResult
from app.training.schemas import SplitQueryParams


def _test_df(uids: list[str], *, split_source: str = "group_id") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_uid": uids,
            "split": ["test"] * len(uids),
            "split_source": [split_source] * len(uids),
            "Efficiency": [1.0] * len(uids),
        }
    )


def _split(use_original_fold: bool = False) -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_2",
        train_pct=0.8,
        test_pct=0.2,
        use_original_fold=use_original_fold,
    )


def test_overlap_with_recorded_training_loci_is_a_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa", "ps:bbb"},
    )
    result = assess_leakage(
        test_df=_test_df(["ps:bbb", "ps:ccc"]),
        split=_split(),
        model="deepprime",
        weights_id="w1",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_TRAIN_TEST_OVERLAP
    assert result.detail["n_overlap_loci"] == 1
    assert "ps:bbb" in result.detail["example_overlap_target_uids"]


def test_disjoint_recorded_training_loci_is_not_a_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa", "ps:bbb"},
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc", "ps:ddd"]),
        split=_split(),
        model="deepprime",
        weights_id="w1",
    )
    assert result is None


def test_unknown_provenance_synthetic_test_with_original_fold_requested(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="group_id"),
        split=_split(use_original_fold=True),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT


def test_unknown_provenance_author_holdout_is_trusted(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is None


def test_unknown_provenance_synthetic_without_original_fold_is_unverifiable(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="group_id"),
        split=_split(use_original_fold=False),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_UNVERIFIABLE_PROVENANCE


def test_execute_evaluation_emits_parseable_leak_error(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:bbb"},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="some/benchmark",
        weights="w1",
        study="some",
        dataset="benchmark",
    )
    from app.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:bbb", "ps:ccc"]))

    with patch("app.evaluation.runner.fetch_model_format_result", return_value=fetch):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["status"] == "error"
    assert result["error_type"] == "data_leak"
    assert result["leak_reason"] == REASON_TRAIN_TEST_OVERLAP
    assert result["metrics"] is None

    manifest = get_job(job_id)
    assert manifest["status"] == "failed"
    assert manifest["error"].startswith("data_leak:")
    assert manifest["result"]["error_type"] == "data_leak"


def test_execute_evaluation_allow_data_leak_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:bbb"},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="some/benchmark",
        weights="w1",
        study="some",
        dataset="benchmark",
        allow_data_leak=True,
    )
    from app.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:bbb", "ps:ccc"]))

    class _StubModel:
        def evaluate(self, test_df, weights):
            return {"pearson": 0.5, "n_samples": len(test_df)}

    with patch("app.evaluation.runner.fetch_model_format_result", return_value=fetch), patch(
        "app.evaluation.runner.ModelFactory.create_model", return_value=_StubModel()
    ):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["metrics"] is not None
    assert result["leak_warning"]["reason"] == REASON_TRAIN_TEST_OVERLAP
    manifest = get_job(job_id)
    assert manifest["status"] == "succeeded"

"""Tests for automatic benchmark resolution from training metadata."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.evaluation.benchmark import (
    BenchmarkResolutionError,
    benchmark_label_from_training,
    resolve_evaluation_request,
)
from app.evaluation.schemas import EvaluationRequest
from app.training.schemas import SplitQueryParams


def _training_metadata() -> dict:
    return {
        "dataset_name": "deepprime/library2",
        "filters": {
            "study": ["deepprime"],
            "dataset": ["library2"],
            "cell_line": ["hek293t"],
            "pe_system": ["pe2"],
        },
        "split": {
            "split_strategy": "holdout_3",
            "train_pct": 0.7,
            "val_pct": 0.15,
            "test_pct": 0.15,
            "use_original_fold": False,
            "split_random_state": 99,
            "merge": False,
        },
    }


def test_benchmark_label_from_training_prefers_dataset_name():
    label = benchmark_label_from_training({"dataset_name": "my-run"})
    assert label == "my-run"


def test_benchmark_label_from_training_builds_from_filters():
    label = benchmark_label_from_training(
        {"filters": {"study": ["deepprime"], "dataset": ["library2"]}}
    )
    assert label == "deepprime/library2"


def test_resolve_uses_training_filters_and_split():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="trained-weights",
    )
    with patch(
        "app.evaluation.benchmark.weights_registry.load_training_metadata",
        return_value=_training_metadata(),
    ):
        resolved = resolve_evaluation_request(request)

    assert resolved.benchmark_name == "deepprime/library2"
    assert resolved.study == ["deepprime"]
    assert resolved.dataset == ["library2"]
    assert resolved.cell_line == ["hek293t"]
    assert resolved.split.split_strategy == "holdout_3"
    assert resolved.split.split_random_state == 99


def test_resolve_preserves_explicit_filter_overrides():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="trained-weights",
        benchmark_name="custom-label",
        dataset=["other"],
    )
    with patch(
        "app.evaluation.benchmark.weights_registry.load_training_metadata",
        return_value=_training_metadata(),
    ):
        resolved = resolve_evaluation_request(request)

    assert resolved.benchmark_name == "custom-label"
    assert resolved.dataset == ["other"]
    # Auto mode always reuses the training split so the test partition matches.
    assert resolved.split.split_strategy == "holdout_3"
    assert resolved.split.split_random_state == 99


def test_manual_mode_preserves_custom_split():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="trained-weights",
        auto_training_benchmark=False,
        benchmark_name="custom",
        split=SplitQueryParams(split_strategy="holdout_2", train_pct=0.8, test_pct=0.2),
    )
    with patch(
        "app.evaluation.benchmark.weights_registry.load_training_metadata",
        return_value=_training_metadata(),
    ):
        resolved = resolve_evaluation_request(request)

    assert resolved.split.split_strategy == "holdout_2"


def test_resolve_requires_manual_filters_for_vendor_weights():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="DeepPrime_base",
    )
    with patch(
        "app.evaluation.benchmark.weights_registry.load_training_metadata",
        return_value=None,
    ):
        with pytest.raises(BenchmarkResolutionError, match="no recorded training metadata"):
            resolve_evaluation_request(request)


def test_resolve_allows_vendor_weights_with_manual_filters():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="DeepPrime_base",
        study=["deepprime"],
        dataset=["library2"],
    )
    with patch(
        "app.evaluation.benchmark.weights_registry.load_training_metadata",
        return_value=None,
    ):
        resolved = resolve_evaluation_request(request)

    assert resolved.benchmark_name == "custom-benchmark"
    assert resolved.study == ["deepprime"]


def test_resolve_disabled_requires_benchmark_name():
    request = EvaluationRequest(
        model_name="deepprime",
        weights="w1",
        auto_training_benchmark=False,
    )
    with pytest.raises(BenchmarkResolutionError, match="benchmark_name is required"):
        resolve_evaluation_request(request)

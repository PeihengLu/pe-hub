"""Resolve evaluation benchmarks from recorded training metadata.

When a weight set was trained in this service, its manifest stores the dataset
filters and split configuration used for training. Evaluating on the matching
held-out test partition only requires the model name and weight ID — the test
set is reconstructed automatically from that provenance.
"""
from __future__ import annotations

from typing import Any, Optional

from ..models import weights_registry
from ..training.schemas import SplitQueryParams
from .schemas import EvaluationRequest

_FILTER_FIELDS = (
    "study",
    "dataset",
    "cell_line",
    "pe_system",
    "edit_type",
    "edit_length",
    "edit_scope",
    "experimental_method",
    "target_context",
    "scaffold_name",
)


class BenchmarkResolutionError(ValueError):
    """Raised when an evaluation benchmark cannot be resolved."""


def _first_filter_value(filters: dict[str, Any], key: str) -> Optional[str]:
    raw = filters.get(key)
    if raw is None:
        return None
    if isinstance(raw, list):
        return str(raw[0]) if raw else None
    return str(raw)


def benchmark_label_from_training(training: dict[str, Any]) -> str:
    """Build a human-readable benchmark label from training metadata."""
    name = training.get("dataset_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    filters = training.get("filters") or {}
    study = _first_filter_value(filters, "study")
    dataset = _first_filter_value(filters, "dataset")
    if study and dataset:
        return f"{study}/{dataset}"
    if dataset:
        return dataset
    return "training-benchmark"


def _request_has_manual_filters(request: EvaluationRequest) -> bool:
    if request.records:
        return True
    return any(getattr(request, field) is not None for field in _FILTER_FIELDS)


def resolve_evaluation_request(request: EvaluationRequest) -> EvaluationRequest:
    """Fill benchmark filters and split from training metadata when enabled.

    When ``auto_training_benchmark`` is True and the weight set records training
    provenance, the evaluation uses the same PE-DB filters and split assignment
    as training so the test partition is the held-out data from that run.

    Explicit request fields always win over training defaults (partial override).
    """
    if not request.auto_training_benchmark:
        if not request.benchmark_name:
            raise BenchmarkResolutionError(
                "benchmark_name is required when auto_training_benchmark is disabled"
            )
        return request

    training = weights_registry.load_training_metadata(
        request.model_name.strip().lower(),
        request.weights.strip(),
    )
    if training is None:
        if not _request_has_manual_filters(request):
            raise BenchmarkResolutionError(
                "Weight set has no recorded training metadata; provide benchmark "
                "filters (study, dataset, …) or train the model via PE-Ensemble "
                "so the held-out test set can be selected automatically."
            )
        benchmark_name = request.benchmark_name or "custom-benchmark"
        return request.model_copy(update={"benchmark_name": benchmark_name})

    updates: dict[str, Any] = {}
    filters = training.get("filters") or {}
    for field in _FILTER_FIELDS:
        if getattr(request, field) is None and field in filters:
            updates[field] = filters[field]

    split_data = training.get("split")
    if isinstance(split_data, dict):
        # Reuse the exact split config from training (fractions, random_state,
        # use_original_fold, merge, …) so the test partition matches training.
        updates["split"] = SplitQueryParams.model_validate(split_data)

    benchmark_name = request.benchmark_name or benchmark_label_from_training(training)
    updates["benchmark_name"] = benchmark_name
    return request.model_copy(update=updates)

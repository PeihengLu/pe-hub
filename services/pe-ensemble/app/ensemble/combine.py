"""Fuse member model predictions without retraining base models."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import rankdata

CombineMethod = Literal[
    "mean",
    "weighted_mean", # user provided weights
    "median",
    "trimmed_mean",
    "rank_mean",
    "percentile_mean",
    "geometric_mean",
    "harmonic_mean",
    "min",
    "max",
]

COMBINE_METHODS: tuple[CombineMethod, ...] = (
    "mean",
    "weighted_mean",
    "median",
    "trimmed_mean",
    "rank_mean",
    "percentile_mean",
    "geometric_mean",
    "harmonic_mean",
    "min",
    "max",
)

_EPSILON = 1e-8


def validate_combine_method(method: str) -> CombineMethod:
    normalized = method.strip().lower()
    if normalized not in COMBINE_METHODS:
        raise ValueError(
            f"Invalid combine method {method!r}. "
            f"Choose one of: {', '.join(COMBINE_METHODS)}"
        )
    return normalized  # type: ignore[return-value]


def _normalize_member_weights(
    weights: Sequence[float],
    *,
    n_members: int,
) -> np.ndarray:
    if len(weights) != n_members:
        raise ValueError(f"weighted_mean requires {n_members} member weights, got {len(weights)}")
    array = np.asarray(weights, dtype=float)
    if np.any(array < 0):
        raise ValueError("Member weights must be non-negative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Member weights must sum to a positive value")
    return array / total


def _percentile_scores(column: np.ndarray) -> np.ndarray:
    n = column.size
    if n <= 1:
        return np.zeros_like(column, dtype=float)
    ranks = rankdata(column, method="average")
    return (ranks - 0.5) / n


def _inverse_percentile_mapping(
    percentile: float,
    sorted_column: np.ndarray,
) -> float:
    n = sorted_column.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_column[0])
    quantile_grid = np.linspace(0.0, 1.0, n)
    return float(np.interp(percentile, quantile_grid, sorted_column))


def combine_predictions(
    predictions: np.ndarray,
    method: CombineMethod | str,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    Combine a matrix of member predictions.

    Args:
        predictions: Shape ``(n_samples, n_members)``.
        method: Fusion strategy name.
        options: Method-specific options (e.g. ``weights``, ``trim_count``, ``epsilon``).

    Returns:
        Combined predictions with shape ``(n_samples,)``.
    """
    preds = np.asarray(predictions, dtype=float)
    if preds.ndim != 2:
        raise ValueError("predictions must be a 2D array (n_samples, n_members)")
    if preds.shape[1] < 1:
        raise ValueError("At least one member prediction column is required")

    normalized_method = validate_combine_method(str(method))
    opts: Dict[str, Any] = dict(options or {})
    n_samples, n_members = preds.shape

    if normalized_method == "mean":
        return np.mean(preds, axis=1)

    if normalized_method == "weighted_mean":
        weights = _normalize_member_weights(opts.get("weights", []), n_members=n_members)
        return preds @ weights

    if normalized_method == "median":
        return np.median(preds, axis=1)

    if normalized_method == "trimmed_mean":
        trim_count = int(opts.get("trim_count", 1))
        if trim_count < 0:
            raise ValueError("trim_count must be non-negative")
        if n_members <= 2 * trim_count:
            return np.mean(preds, axis=1)
        sorted_preds = np.sort(preds, axis=1)
        return np.mean(sorted_preds[:, trim_count:-trim_count], axis=1)

    if normalized_method == "rank_mean":
        ranks = np.column_stack(
            [rankdata(preds[:, member_index], method="average") for member_index in range(n_members)]
        )
        avg_rank = ranks.mean(axis=1)
        pooled = np.sort(preds.reshape(-1))
        quantiles = (avg_rank - 0.5) / max(n_samples, 1)
        quantile_grid = np.linspace(0.0, 1.0, pooled.size)
        return np.interp(quantiles, quantile_grid, pooled)

    if normalized_method == "percentile_mean":
        percentiles = np.column_stack(
            [_percentile_scores(preds[:, member_index]) for member_index in range(n_members)]
        )
        avg_percentile = percentiles.mean(axis=1)
        combined = np.zeros(n_samples, dtype=float)
        sorted_columns = [np.sort(preds[:, member_index]) for member_index in range(n_members)]
        for row_index, percentile in enumerate(avg_percentile):
            mapped = [
                _inverse_percentile_mapping(float(percentile), sorted_columns[member_index])
                for member_index in range(n_members)
            ]
            combined[row_index] = float(np.mean(mapped))
        return combined

    epsilon = float(opts.get("epsilon", _EPSILON))
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if normalized_method == "geometric_mean":
        clipped = np.clip(preds, epsilon, None)
        return np.exp(np.mean(np.log(clipped), axis=1))

    if normalized_method == "harmonic_mean":
        clipped = np.clip(preds, epsilon, None)
        return n_members / np.sum(1.0 / clipped, axis=1)

    if normalized_method == "min":
        return np.min(preds, axis=1)

    if normalized_method == "max":
        return np.max(preds, axis=1)

    raise ValueError(f"Unsupported combine method: {normalized_method}")


def combine_method_help() -> List[Dict[str, str]]:
    """Short descriptions for API docs and UI tooltips."""
    return [
        {"id": "mean", "label": "Mean", "description": "Unweighted arithmetic average."},
        {
            "id": "weighted_mean",
            "label": "Weighted mean",
            "description": "Weighted average; supply member weights in combine_options.",
        },
        {"id": "median", "label": "Median", "description": "Per-sample median across members."},
        {
            "id": "trimmed_mean",
            "label": "Trimmed mean",
            "description": "Drop lowest/highest member per sample (trim_count, default 1).",
        },
        {
            "id": "rank_mean",
            "label": "Rank mean",
            "description": "Average cross-sample ranks, mapped back to the prediction scale.",
        },
        {
            "id": "percentile_mean",
            "label": "Percentile mean",
            "description": "Average within-batch percentiles, then map back per member.",
        },
        {
            "id": "geometric_mean",
            "label": "Geometric mean",
            "description": "Geometric average (predictions clipped to epsilon).",
        },
        {
            "id": "harmonic_mean",
            "label": "Harmonic mean",
            "description": "Harmonic average (predictions clipped to epsilon).",
        },
        {"id": "min", "label": "Minimum", "description": "Per-sample minimum across members."},
        {"id": "max", "label": "Maximum", "description": "Per-sample maximum across members."},
    ]

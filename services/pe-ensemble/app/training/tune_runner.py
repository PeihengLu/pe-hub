"""Run Optuna trials on top of the shared training runner."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .runner import TrainingError, execute_training
from .schemas import TrainingRequest
from .search_spaces import SearchSpaceSpec, get_search_space


@dataclass(frozen=True)
class TrialResult:
    metric: float
    hyperparameters: Dict[str, Any]
    train_result: Dict[str, Any]


def extract_validation_metric(model_name: str, train_payload: Mapping[str, Any]) -> float:
    """Map a training payload to a scalar objective for Optuna."""
    name = model_name.strip().lower()
    result = train_payload.get("result")
    if not isinstance(result, dict):
        return float("nan")

    if name == "oped":
        value = result.get("val_spearman")
        if value is None and isinstance(result.get("cross_validation"), dict):
            value = result["cross_validation"].get("mean_val_spearman")
        return float(value) if value is not None else float("nan")

    if name == "deepprime":
        summaries = result.get("model_summaries") or []
        if summaries:
            loss = summaries[0].get("best_val_loss")
            if loss is not None and not math.isnan(float(loss)):
                return -float(loss)
        final_val_loss = result.get("final_val_loss")
        if final_val_loss is not None:
            return -float(final_val_loss)
        return float("nan")

    if name == "pridict2":
        metrics = result.get("validation_metrics") or {}
        if not isinstance(metrics, dict):
            return float("nan")
        for key in ("averageedited_spearman", "averageedited_pearson"):
            if key in metrics:
                return float(metrics[key])
        spearman_values = [
            float(value)
            for metric_key, value in metrics.items()
            if metric_key.endswith("_spearman") and value is not None
        ]
        if spearman_values:
            return max(spearman_values)
        return float("nan")

    return float("nan")


def metric_for_search_space(space: SearchSpaceSpec, metric_value: float) -> float:
    if space.direction == "minimize":
        return metric_value
    return metric_value


def run_tuning_trial(
    request: TrainingRequest,
    *,
    suggested: Mapping[str, Any],
    register_weights: bool = False,
) -> TrialResult:
    """Execute one tuning trial with suggested hyperparameters."""
    trial_request = request.model_copy(
        update={
            "hyperparameters": dict(suggested),
            "hyperparameter_mode": "replace",
            "notes": (request.notes or "optuna trial"),
        }
    )
    payload = execute_training(
        trial_request,
        device_id=request.device,
        register_weights=register_weights,
    )
    metric = extract_validation_metric(request.model_name, payload)
    if math.isnan(metric):
        raise TrainingError("Trial completed without a usable validation metric.")
    space = get_search_space(request.model_name)
    return TrialResult(
        metric=metric_for_search_space(space, metric),
        hyperparameters=dict(suggested),
        train_result=payload,
    )

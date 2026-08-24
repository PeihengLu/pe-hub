"""Run Optuna trials for automatic hyperparameter tuning."""
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


def _cv_block(result: Mapping[str, Any]) -> Any:
    return result.get("cross_validation")


def _mean_numeric(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _oped_cv_spearman(result: Mapping[str, Any]) -> float:
    cv_block = _cv_block(result)
    if isinstance(cv_block, dict):
        value = cv_block.get("mean_val_spearman")
        if value is not None and not math.isnan(float(value)):
            return float(value)
        folds = cv_block.get("folds") or []
        scores = [
            float(fold["val_spearman"])
            for fold in folds
            if isinstance(fold, dict) and fold.get("val_spearman") is not None
        ]
        return _mean_numeric(scores)
    if isinstance(cv_block, list):
        scores = [
            float(fold["val_spearman"])
            for fold in cv_block
            if isinstance(fold, dict) and fold.get("val_spearman") is not None
        ]
        return _mean_numeric(scores)
    return float("nan")


def _deepprime_cv_neg_loss(result: Mapping[str, Any]) -> float:
    cv_block = _cv_block(result)
    losses: list[float] = []
    if isinstance(cv_block, dict):
        value = cv_block.get("mean_best_val_loss")
        if value is not None and not math.isnan(float(value)):
            return -float(value)
        folds = cv_block.get("folds") or []
        losses = [
            float(fold["best_val_loss"])
            for fold in folds
            if isinstance(fold, dict) and fold.get("best_val_loss") is not None
        ]
    elif isinstance(cv_block, list):
        losses = [
            float(fold["best_val_loss"])
            for fold in cv_block
            if isinstance(fold, dict) and fold.get("best_val_loss") is not None
        ]
    if not losses:
        return float("nan")
    return -_mean_numeric(losses)


def _pridict2_cv_spearman(result: Mapping[str, Any]) -> float:
    cv_reports = _cv_block(result)
    if not isinstance(cv_reports, list) or not cv_reports:
        return float("nan")
    fold_scores: list[float] = []
    for report in cv_reports:
        fold_metrics = report.get("metrics") if isinstance(report, dict) else None
        if not isinstance(fold_metrics, dict):
            continue
        for key in ("averageedited_spearman", "averageedited_pearson"):
            if key in fold_metrics and fold_metrics[key] is not None:
                fold_scores.append(float(fold_metrics[key]))
                break
    return _mean_numeric(fold_scores)


def extract_validation_metric(model_name: str, train_payload: Mapping[str, Any]) -> float:
    """Map a training payload to a scalar Optuna objective.

    When a trial ran k-fold CV, prefer the mean fold metric over the final
    holdout split so DeepPrime, OPED, and PRIDICT2 tune on the same protocol.
    """
    name = model_name.strip().lower()
    result = train_payload.get("result")
    if not isinstance(result, dict):
        return float("nan")

    if name == "oped":
        cv_metric = _oped_cv_spearman(result)
        if not math.isnan(cv_metric):
            return cv_metric
        value = result.get("val_spearman")
        return float(value) if value is not None else float("nan")

    if name == "deepprime":
        cv_metric = _deepprime_cv_neg_loss(result)
        if not math.isnan(cv_metric):
            return cv_metric
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
        cv_metric = _pridict2_cv_spearman(result)
        if not math.isnan(cv_metric):
            return cv_metric
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

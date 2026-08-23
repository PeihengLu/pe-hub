"""Stable import path for PE Ensemble library functions."""
from __future__ import annotations

from pe_ensemble._bootstrap import import_service_app

_runner = import_service_app("training.runner")
_eval = import_service_app("evaluation.runner")
_ensemble = import_service_app("ensemble.runner")
_tune = import_service_app("training.tune_study")
_config = import_service_app("training.config")
_combine = import_service_app("ensemble.combine")
_registry = import_service_app("models.registry")

TrainingError = _runner.TrainingError
execute_training = _runner.execute_training

EvaluationError = _eval.EvaluationError
execute_evaluation = _eval.execute_evaluation

EnsembleError = _ensemble.EnsembleError
execute_ensemble = _ensemble.execute_ensemble

execute_tuning = _tune.execute_tuning

supported_models = _config.supported_models
is_supported_model = _config.is_supported_model

COMBINE_METHODS = _combine.COMBINE_METHODS
combine_method_help = _combine.combine_method_help

model_registry = _registry.model_registry

__all__ = [
    "COMBINE_METHODS",
    "EnsembleError",
    "EvaluationError",
    "TrainingError",
    "combine_method_help",
    "execute_ensemble",
    "execute_evaluation",
    "execute_training",
    "execute_tuning",
    "is_supported_model",
    "model_registry",
    "supported_models",
]

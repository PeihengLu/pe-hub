"""Headless PE Ensemble API (shared by HTTP handlers and the peen CLI)."""
from __future__ import annotations

from pe_ensemble.ensemble.combine import COMBINE_METHODS, combine_method_help
from pe_ensemble.ensemble.runner import EnsembleError, execute_ensemble
from pe_ensemble.evaluation.runner import EvaluationError, execute_evaluation
from pe_ensemble.models.registry import model_registry
from pe_ensemble.training.config import is_supported_model, supported_models
from pe_ensemble.training.runner import TrainingError, execute_training
from pe_ensemble.training.tune_study import execute_tuning

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

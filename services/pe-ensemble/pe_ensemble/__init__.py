"""Installable PE Ensemble package (FastAPI service, CLI, and in-process library).

``pe_ensemble.library`` is the stable import path for headless use. The HTTP
app is ``pe_ensemble.main:app``.
"""
from __future__ import annotations

from pe_ensemble.library import (
    COMBINE_METHODS,
    EnsembleError,
    EvaluationError,
    TrainingError,
    combine_method_help,
    execute_ensemble,
    execute_evaluation,
    execute_training,
    execute_tuning,
    is_supported_model,
    model_registry,
    supported_models,
)

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

__version__ = "0.2.0"

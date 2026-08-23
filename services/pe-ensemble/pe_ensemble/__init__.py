"""Installable PE Ensemble library package (CLI + in-process access).

The FastAPI service lives under ``app/``; this namespace is the stable import
path for headless use on clusters and from other Python code.
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

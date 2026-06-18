"""Model architecture (size) hyperparameters for training UI and CLI."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..models.registry import model_registry


def build_architecture_hyperparameters(
    model_name: str,
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map UI/CLI architecture fields to wrapper hyperparameter keys."""
    return model_registry.build_architecture(model_name, values)


def merge_training_hyperparameters(
    model_name: str,
    base: Optional[Dict[str, Any]],
    architecture: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge architecture fields into an existing hyperparameters dict."""
    merged = dict(base or {})
    if not merged.get("load_pretrained"):
        merged.update(build_architecture_hyperparameters(model_name, architecture))
    return apply_fine_tune_defaults(merged)


def apply_fine_tune_defaults(hyperparameters: Mapping[str, Any]) -> Dict[str, Any]:
    """When fine-tuning from pretrained weights, freeze the representation backbone."""
    merged = dict(hyperparameters)
    if merged.get("load_pretrained"):
        merged["freezing"] = True
    return merged


def architecture_from_cli_args(model_name: str, args: Any) -> Dict[str, Any]:
    """Read optional architecture CLI flags for the selected model."""
    return model_registry.architecture_from_cli(model_name, args)

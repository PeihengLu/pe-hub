"""Shared hyperparameter helpers for PE Ensemble wrappers.

Vendor wire names stay per model (``epoch_num`` vs ``epochs`` vs ``num_epochs``).
These helpers only read aliases and resolve pretrained weight IDs; they do not
unify Lightning and JAX training loops.
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Optional

import pandas as pd

from pe_common.splits import has_assigned_cv_folds, iter_assigned_cv_folds


def resolve_pretrained_weight_id(
    hyperparameters: Optional[Mapping[str, Any]],
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return a weight ID when ``load_pretrained`` is set.

    ``weights`` wins when present and non-blank. Callers pass their vendor
    default, or omit ``default`` so a missing ID means "use the wrapper's
    ``load_model()`` path" (DeepPrime) or "skip the statedict" (PRIDICT2).
    """
    if not hyperparameters or not bool(hyperparameters.get("load_pretrained", False)):
        return None
    weights = hyperparameters.get("weights")
    if weights is not None and str(weights).strip():
        return str(weights)
    return default


def require_evaluate_weights(wrapper: Any, weights: str) -> str:
    """Reject a blank evaluate() weight ID with the wrapper's available list."""
    if not weights or not str(weights).strip():
        raise ValueError(
            "weights is required for evaluate(). "
            f"Available: {wrapper.list_available_weights()}"
        )
    return str(weights)


def iter_cv_training_folds(
    df: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Iterator[tuple[int, str, pd.DataFrame, pd.DataFrame]]:
    """Yield assigned CV folds when the caller did not pass an explicit val frame."""
    if val_data is not None:
        return
    if not has_assigned_cv_folds(df):
        return
    for fold_idx, (fold_label, fold_train, fold_val) in enumerate(
        iter_assigned_cv_folds(df)
    ):
        if cancel_check is not None:
            cancel_check()
        yield fold_idx, fold_label, fold_train, fold_val

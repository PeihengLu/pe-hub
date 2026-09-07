"""Tests for wrapper hyperparameter aliases and pretrained-weight resolution."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from pe_ensemble.models.hparams import (
    iter_cv_training_folds,
    require_evaluate_weights,
    resolve_pretrained_weight_id,
)


def test_resolve_pretrained_weight_id_requires_flag():
    assert resolve_pretrained_weight_id({"weights": "ckpt"}) is None
    assert resolve_pretrained_weight_id({"load_pretrained": False, "weights": "ckpt"}) is None


def test_resolve_pretrained_weight_id_prefers_explicit_weights():
    assert (
        resolve_pretrained_weight_id(
            {"load_pretrained": True, "weights": "ckpt-a"},
            default="vendor-default",
        )
        == "ckpt-a"
    )


def test_resolve_pretrained_weight_id_uses_caller_default():
    assert (
        resolve_pretrained_weight_id({"load_pretrained": True}, default="vendor-default")
        == "vendor-default"
    )
    assert resolve_pretrained_weight_id({"load_pretrained": True}) is None
    assert resolve_pretrained_weight_id({"load_pretrained": True, "weights": "  "}) is None


def test_require_evaluate_weights_rejects_blank():
    wrapper = SimpleNamespace(list_available_weights=lambda: ["a", "b"])
    with pytest.raises(ValueError, match="weights is required"):
        require_evaluate_weights(wrapper, "")
    assert require_evaluate_weights(wrapper, "a") == "a"


def test_iter_cv_training_folds_skips_explicit_val():
    df = pd.DataFrame({"split": ["fold_0", "fold_1"], "x": [1, 2]})
    assert list(iter_cv_training_folds(df, val_data=df)) == []


def test_iter_cv_training_folds_yields_assigned_folds():
    df = pd.DataFrame(
        {
            "split": ["fold_0", "fold_0", "fold_1", "fold_1"],
            "x": [1, 2, 3, 4],
        }
    )
    calls = []
    folds = list(iter_cv_training_folds(df, cancel_check=lambda: calls.append(1)))
    assert len(folds) == 2
    assert calls == [1, 1]
    assert [label for _, label, _, _ in folds] == ["fold_0", "fold_1"]
    assert list(folds[0][2]["x"]) == [3, 4]
    assert list(folds[0][3]["x"]) == [1, 2]

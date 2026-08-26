"""Tests for training architecture hyperparameter helpers."""
from __future__ import annotations

from types import SimpleNamespace

from app.training.model_architecture import (
    apply_fine_tune_defaults,
    architecture_from_cli_args,
    build_architecture_hyperparameters,
    merge_training_hyperparameters,
)


def test_build_oped_architecture_expands_branch_dims():
    out = build_architecture_hyperparameters(
        "oped",
        {
            "embedding_size": 32,
            "ffn_dim": 512,
            "encoder_layers": 2,
            "nhead": 4,
            "dropout": 0.2,
        },
    )
    assert out == {
        "embedding_size": 32,
        "hidden_size": [512, 512, 512],
        "num_encoder_layers": [2, 2, 2],
        "nhead": 4,
        "drop_out": 0.2,
    }


def test_build_pridict2_architecture_keys():
    out = build_architecture_hyperparameters(
        "pridict2",
        {"embed_dim": 128, "num_hidden_layers": 2, "dropout": 0.15},
    )
    assert out == {
        "embed_dim": 128,
        "num_hidden_layers": 2,
        "p_dropout": 0.15,
    }


def test_merge_training_hyperparameters_preserves_existing():
    merged = merge_training_hyperparameters(
        "deepprime",
        {"lr": 1e-4, "epochs": 5},
        {"hidden_size": 256, "num_layers": 2},
    )
    assert merged["lr"] == 1e-4
    assert merged["epochs"] == 5
    assert merged["hidden_size"] == 256
    assert merged["num_layers"] == 2


def test_apply_fine_tune_defaults_freezes_backbone():
    out = apply_fine_tune_defaults({"load_pretrained": True, "lr": 1e-4})
    assert out["freezing"] is True
    assert out["lr"] == 1e-4


def test_apply_fine_tune_defaults_leaves_scratch_training():
    out = apply_fine_tune_defaults({"epochs": 5})
    assert "freezing" not in out


def test_merge_training_hyperparameters_skips_architecture_when_finetuning():
    merged = merge_training_hyperparameters(
        "deepprime",
        {"lr": 1e-4, "load_pretrained": True, "weights": "DeepPrime_base"},
        {"hidden_size": 256, "num_layers": 2},
    )
    assert merged["lr"] == 1e-4
    assert merged["load_pretrained"] is True
    assert merged["freezing"] is True
    assert "hidden_size" not in merged
    assert "num_layers" not in merged


def test_architecture_from_cli_args_model_specific():
    args = SimpleNamespace(
        dp_hidden_size=96,
        dp_num_layers=2,
        oped_embedding_size=None,
        oped_ffn_dim=None,
        oped_encoder_layers=None,
        oped_nhead=None,
        oped_dropout=None,
        pridict2_embed_dim=None,
        pridict2_num_hidden_layers=None,
        pridict2_dropout=None,
    )
    assert architecture_from_cli_args("deepprime", args) == {
        "hidden_size": 96,
        "num_layers": 2,
    }

"""Wrapper-aligned baseline hyperparameters (no dataset presets)."""
from __future__ import annotations

from typing import Any, Dict


def model_baseline_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Return model-level fallback defaults when no preset is available."""
    name = model_name.strip().lower()
    if name == "deepprime":
        return {
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0,
            "reshuffle_each_epoch": True,
            "load_pretrained": False,
            "hidden_size": 128,
            "num_layers": 1,
        }
    if name == "oped":
        return {
            "epoch_num": 100,
            "batch_size": 128,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0,
            "reshuffle_each_epoch": True,
            "embedding_size": 64,
            "hidden_size": [2048, 2048, 2048],
            "num_encoder_layers": [6, 6, 6],
            "nhead": 8,
            "drop_out": 0.1,
        }
    if name == "pridict2":
        return {
            "batch_size": 128,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 20,
            "embed_dim": 64,
            "num_hidden_layers": 1,
            "p_dropout": 0.1,
            "loss_func": "MSEloss",
            "y_ref": ["averageedited"],
        }
    return {}

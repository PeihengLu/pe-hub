"""Tests for PRIDICT2 Lightning training helpers."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from app.models.pridict2_lightning import (
    _PRIDICT2LightningModule,
    build_pernn_distribution_model,
    build_pridict_loss,
)


def test_build_pridict_loss_kld():
    loss = build_pridict_loss("KLDloss")
    assert isinstance(loss, torch.nn.KLDivLoss)


def test_pridict2_lightning_configure_optimizers_with_scheduler():
    model = build_pernn_distribution_model(
        {
            "embed_dim": 16,
            "num_hidden_layers": 1,
            "p_dropout": 0.1,
            "annot_embed": 4,
            "assemb_opt": "add",
            "lr": 0.001,
            "weight_decay": 1e-5,
            "scheduler": "step",
            "scheduler_kwargs": {"step_size": 5, "gamma": 0.9},
        },
        seqlevel_featdim=3,
        num_outcomes=3,
        device=torch.device("cpu"),
    )
    module = _PRIDICT2LightningModule(
        model,
        train_hparams={
            "lr": 0.001,
            "weight_decay": 1e-5,
            "scheduler": "step",
            "scheduler_kwargs": {"step_size": 5, "gamma": 0.9},
        },
        loss_func_name="KLDloss",
        device=torch.device("cpu"),
    )
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config

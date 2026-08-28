"""Tests for PRIDICT2 Lightning training helpers."""
from __future__ import annotations

import torch

from app.models.pridict2_wrapper import (
    build_pernn_distribution_model,
    build_pridict_loss,
    _PRIDICT2LightningModule,
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
            "lr": 0.001,
            "weight_decay": 1e-5,
            "scheduler": "step",
            "scheduler_kwargs": {"step_size": 5, "gamma": 0.9},
        },
        seqlevel_featdim=3,
        num_outcomes=1,
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
        loss_func_name="MSEloss",
    )
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config


def test_pridict2_stack_assembly_accepts_mismatched_embed_widths():
    """Assembly is fixed to stack; HPO may vary embed_dim with annot_embed=8."""
    model = build_pernn_distribution_model(
        {"embed_dim": 128},
        seqlevel_featdim=3,
        num_outcomes=1,
        device=torch.device("cpu"),
    )
    batch, seq_len = 2, 10
    nucl = torch.zeros(batch, seq_len, dtype=torch.long)
    proto = torch.zeros(batch, seq_len, dtype=torch.long)
    pbs = torch.zeros(batch, seq_len, dtype=torch.long)
    rt = torch.zeros(batch, seq_len, dtype=torch.long)
    init = model.init_annot_embed(nucl, proto, pbs, rt)
    mut = model.mut_annot_embed(nucl, pbs, rt)
    assert init.shape == (batch, seq_len, 128 + 3 * 8)
    assert mut.shape == (batch, seq_len, 128 + 2 * 8)
    # Hyperparameter overrides for assemb_opt / annot_embed are ignored.
    ignored = build_pernn_distribution_model(
        {"embed_dim": 32, "assemb_opt": "add", "annot_embed": 4},
        seqlevel_featdim=3,
        num_outcomes=1,
        device=torch.device("cpu"),
    )
    assert ignored.init_annot_embed.assemb_opt == "stack"
    assert ignored.init_annot_embed.Wproto.embedding_dim == 8

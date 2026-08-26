"""Tests for PRIDICT2 Lightning training helpers."""
from __future__ import annotations

from pathlib import Path

import torch

from app.models.pridict2_wrapper import (
    PERNNDistributionModel,
    _PRIDICT2LightningModule,
    build_pernn_distribution_model,
    build_pridict_loss,
)


def test_build_pridict_loss_kld():
    loss = build_pridict_loss("KLDloss")
    assert isinstance(loss, torch.nn.KLDivLoss)


def test_resolve_training_outcomes_rejects_kld_without_distribution():
    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper
    import pandas as pd

    df = pd.DataFrame({"averageedited": [10.0, 20.0]})
    try:
        PRIDICT2ModelWrapper._resolve_training_outcomes(
            df,
            requested=["averageedited"],
            loss_func="KLDloss",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "averageunedited" in str(exc)


def test_resolve_training_outcomes_rejects_kld_with_partial_nan_trio():
    """L1+ClinVar merge materializes trio columns but ClinVar rows are NaN."""
    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper
    import pandas as pd

    df = pd.DataFrame(
        {
            "averageedited": [10.0, 5.0],
            "averageunedited": [80.0, float("nan")],
            "averageindel": [10.0, float("nan")],
        }
    )
    try:
        PRIDICT2ModelWrapper._resolve_training_outcomes(
            df,
            requested=["averageedited", "averageunedited", "averageindel"],
            loss_func="KLDloss",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "non-null" in str(exc)
        assert "MSEloss" in str(exc)


def test_resolve_training_outcomes_mse_uses_averageedited_only():
    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper
    import pandas as pd

    df = pd.DataFrame(
        {
            "averageedited": [10.0, 5.0],
            "averageunedited": [80.0, float("nan")],
            "averageindel": [10.0, float("nan")],
        }
    )
    outcomes = PRIDICT2ModelWrapper._resolve_training_outcomes(
        df,
        requested=["averageedited", "averageunedited", "averageindel"],
        loss_func="MSEloss",
    )
    assert outcomes == ["averageedited"]


def test_resolve_training_outcomes_kld_uses_full_distribution():
    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper
    import pandas as pd

    df = pd.DataFrame(
        {
            "averageedited": [10.0],
            "averageunedited": [80.0],
            "averageindel": [10.0],
        }
    )
    outcomes = PRIDICT2ModelWrapper._resolve_training_outcomes(
        df,
        requested=["averageedited"],
        loss_func="KLDloss",
    )
    assert outcomes == ["averageedited", "averageunedited", "averageindel"]


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
    )
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config


def test_pridict2_stack_assembly_accepts_mismatched_embed_widths():
    """Assembly is fixed to stack; HPO may vary embed_dim with annot_embed=8."""
    model = build_pernn_distribution_model(
        {"embed_dim": 128},
        seqlevel_featdim=3,
        num_outcomes=3,
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
        num_outcomes=3,
        device=torch.device("cpu"),
    )
    assert ignored.init_annot_embed.assemb_opt == "stack"
    assert ignored.init_annot_embed.Wproto.embedding_dim == 8


def test_flexible_pretrained_load_skips_mismatched_decoder(tmp_path: Path):
    source = build_pernn_distribution_model(
        {"embed_dim": 16, "num_hidden_layers": 1, "p_dropout": 0.1},
        seqlevel_featdim=3,
        num_outcomes=3,
        device=torch.device("cpu"),
    )
    target = build_pernn_distribution_model(
        {"embed_dim": 16, "num_hidden_layers": 1, "p_dropout": 0.1},
        seqlevel_featdim=3,
        num_outcomes=1,
        device=torch.device("cpu"),
    )
    statedict_dir = tmp_path / "model_statedict"
    source.save_vendor_statedict(str(statedict_dir))

    before_decoder = target.decoder.state_dict()["We.weight"].clone()
    summary = target.load_vendor_statedict_flexible(
        str(statedict_dir),
        device=torch.device("cpu"),
    )

    assert "decoder" in summary["skipped_mismatched"]
    assert "init_encoder" in summary["loaded"]
    assert "seqlevel_featembeder" in summary["loaded"]
    assert torch.equal(before_decoder, target.decoder.state_dict()["We.weight"])
    for name in PERNNDistributionModel.COMPONENT_NAMES:
        if name == "decoder":
            continue
        source_state = dict(source.iter_components())[name].state_dict()
        target_state = dict(target.iter_components())[name].state_dict()
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            assert torch.equal(source_state[key], target_state[key])


def test_resolve_pretrained_statedict_dir_prefers_ensemble_run_layout(tmp_path: Path, monkeypatch):
    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

    run_dir = tmp_path / "pridict2__hek293t-pe2__smoke"
    (run_dir / "model_statedict").mkdir(parents=True)
    (run_dir / "config").mkdir()

    monkeypatch.setattr(
        PRIDICT2ModelWrapper,
        "resolve_weight_selection",
        staticmethod(lambda name: (run_dir.resolve(), None)),
    )
    resolved = PRIDICT2ModelWrapper._resolve_pretrained_statedict_dir("smoke")
    assert Path(resolved) == (run_dir / "model_statedict").resolve()

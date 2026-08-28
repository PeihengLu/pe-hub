"""Tests for pe_common.training helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pe-common"))

from pe_common.training import (
    LightningTrainerConfig,
    apply_fine_tune_freezing,
    fit_lightning_module,
    format_epoch_metrics_row,
    run_supervised_training_loop,
)


def test_format_epoch_metrics_row():
    line = format_epoch_metrics_row(
        {
            "model_index": 0.0,
            "epoch": 3.0,
            "train_loss": 0.123456,
            "val_loss": 0.234567,
        },
        prefix="final",
    )
    assert line == "final | model 0 | epoch 3 | train_loss=0.123456 | val_loss=0.234567"


class _DummyModel:
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor(1.0)}

    def load_state_dict(self, _state: dict[str, torch.Tensor]) -> None:
        return None


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def test_apply_fine_tune_freezing_trains_head_only():
    model = _ToyModel()
    apply_fine_tune_freezing(model, model.head)
    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert all(param.requires_grad for param in model.head.parameters())


def test_run_supervised_training_loop_calls_on_epoch_end():
    rows: list[dict[str, float]] = []

    def train_epoch(_epoch: int) -> float:
        return 1.0

    def validate_epoch(_epoch: int) -> dict[str, float]:
        return {"val_loss": 0.5}

    run_supervised_training_loop(
        model=_DummyModel(),  # type: ignore[arg-type]
        num_epochs=2,
        train_epoch_fn=train_epoch,
        validate_epoch_fn=validate_epoch,
        on_epoch_end=rows.append,
    )
    assert len(rows) == 2
    assert rows[0]["epoch"] == 0.0
    assert rows[1]["epoch"] == 1.0


class _FitDeviceModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(4, 1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-2)


def test_fit_lightning_module_val_loss_updates_across_epochs():
    rows: list[dict[str, float]] = []
    x = torch.randn(128, 4)
    y = torch.randn(128, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    class _LearningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Linear(4, 1)

        def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
        ) -> torch.Tensor:
            features, targets = batch
            loss = ((self.model(features) - targets) ** 2).mean()
            self.log("train_loss", loss, on_epoch=True, on_step=False)
            return loss

        def validation_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
        ) -> torch.Tensor:
            features, targets = batch
            loss = ((self.model(features) - targets) ** 2).mean()
            self.log("val_loss", loss, on_epoch=True, on_step=False)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.model.parameters(), lr=0.05)

    module = _LearningModule()
    metrics = fit_lightning_module(
        module,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        config=LightningTrainerConfig(max_epochs=4, patience=None, enable_progress_bar=False),
        on_epoch_end=rows.append,
    )
    assert len(rows) == 4
    val_losses = [row["val_loss"] for row in rows]
    assert val_losses[0] != val_losses[-1]
    assert int(metrics["best_epoch"]) >= 0


def test_fit_lightning_module_restores_training_device():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    module = _FitDeviceModule()
    fit_lightning_module(
        module,
        train_loader=loader,
        val_loader=loader,
        device=device,
        config=LightningTrainerConfig(max_epochs=1, patience=None, enable_progress_bar=False),
    )
    assert next(module.model.parameters()).device.type == "cuda"
"""Tests for pe_common.training helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pe-common"))

from pe_common.training import format_epoch_metrics_row, run_supervised_training_loop


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

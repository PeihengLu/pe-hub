"""Default BasePEModel orchestration hooks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from pe_common.model_interface import BasePEModel


class _StubModel(BasePEModel):
    def __init__(self) -> None:
        super().__init__("stub")
        self.prepared: Optional[pd.DataFrame] = None

    def load_model(self, model_path: str) -> None:
        return None

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        self.prepared = df
        return df

    def predict(self, data: Any, batch_size: int = 32) -> List[float]:
        return [0.1] * len(data)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {}

    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        return {}

    def save_model(self, model_path: str) -> None:
        return None


def test_default_hooks_are_identity_and_drop_label_columns():
    model = _StubModel()
    frame = pd.DataFrame({"x": [1, 2], "Efficiency": [0.1, 0.2]})
    assert model.prepare_training_frame(frame) is frame
    assert model.prepare_evaluation_frame(frame) is frame
    assert model.capture_stderr_during_run() is False
    predictions = model.predict_on_frame(frame)
    assert predictions == [0.1, 0.1]
    assert model.prepared is not None
    assert "Efficiency" not in model.prepared.columns
    assert list(model.prepared["x"]) == [1, 2]

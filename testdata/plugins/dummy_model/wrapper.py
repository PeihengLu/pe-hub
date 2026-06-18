"""Minimal BasePEModel wrapper for plugin loader tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pe_common.model_interface import BasePEModel
from pe_common.training import regression_metrics


class DummyModelWrapper(BasePEModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(model_name="dummy_model", device=device)
        self._artifact_path: Optional[Path] = None

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dummy weights not found: {path}")
        self._artifact_path = path
        self.is_trained = True

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.copy()

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> List[float]:
        if "feature" not in data.columns:
            raise ValueError("Dummy model expects a 'feature' column")
        return [float(value) for value in data["feature"].tolist()]

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.is_trained = True
        self._artifact_path = None
        return {"epochs": (hyperparameters or {}).get("epochs", 1)}

    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        self.load_weights_by_name(weights)
        preds = self.predict(self.prepare_data(test_data))
        y_true = test_data["Efficiency"].astype(float).tolist()
        return regression_metrics(y_true, preds)

    def save_model(self, model_path: str) -> None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy-trained-artifact\n", encoding="utf-8")
        self._artifact_path = path

    def save_to_registry(self, dest_dir) -> str:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        artifact = dest / "weights.txt"
        artifact.write_text("dummy-trained-artifact\n", encoding="utf-8")
        self._artifact_path = artifact
        return "dummy_state_dict"

    def load_weights_by_name(self, name: str) -> None:
        from app.models import weights_registry

        entry_dir = weights_registry.resolve_dir("dummy_model", name)
        artifact = entry_dir / "weights.txt"
        if artifact.is_file():
            self.load_model(str(artifact))
            return
        raise FileNotFoundError(f"No dummy weights artifact in {entry_dir}")

    @staticmethod
    def list_available_weights() -> List[str]:
        from app.models import weights_registry

        return weights_registry.list_weight_ids("dummy_model")

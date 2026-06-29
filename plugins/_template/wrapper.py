"""PE Ensemble wrapper for my_model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pe_common.model_interface import BasePEModel
from pe_common.training import regression_metrics

ARTIFACT_NAME = "model.pt"


class MyModelWrapper(BasePEModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(model_name="my_model", device=device)
        self.model = None

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if path.is_dir():
            path = path / ARTIFACT_NAME
        if not path.is_file():
            raise FileNotFoundError(path)
        # torch.load(path, map_location=self.device) ...
        self.is_trained = True

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.copy()

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> List[float]:
        return [float(v) for v in data["feature"].tolist()]

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hp = hyperparameters or {}
        if hp.get("load_pretrained"):
            self.load_weights_by_name(str(hp["weights"]))
        self.is_trained = True
        return {"epochs": int(hp.get("epochs", 10))}

    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        self.load_weights_by_name(weights)
        preds = self.predict(self.prepare_data(test_data))
        y_true = test_data["Efficiency"].astype(float).tolist()
        return regression_metrics(y_true, preds)

    def save_model(self, model_path: str) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).write_text("placeholder\n", encoding="utf-8")

    def save_to_registry(self, dest_dir) -> str:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ARTIFACT_NAME).write_text("placeholder\n", encoding="utf-8")
        return "my_model_weights"

    def load_weights_by_name(self, name: str) -> None:
        from app.models import weights_registry

        entry_dir = weights_registry.resolve_dir("my_model", name)
        self.load_model(str(entry_dir / ARTIFACT_NAME))

    @staticmethod
    def list_available_weights() -> List[str]:
        from app.models import weights_registry

        return weights_registry.list_weight_ids("my_model")

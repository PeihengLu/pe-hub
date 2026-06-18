"""Minimal wrapper used only when convert import fails before wrapper checks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from pe_common.model_interface import BasePEModel


class BrokenConvertWrapper(BasePEModel):
    def load_model(self, model_path: str) -> None:
        self.is_trained = True

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> List[float]:
        return [0.0] * len(data)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {}

    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        return {"pearson": 1.0, "spearman": 1.0, "n_samples": len(test_data)}

    def save_model(self, model_path: str) -> None:
        pass

    def save_to_registry(self, dest_dir) -> str:
        return "dummy_state_dict"

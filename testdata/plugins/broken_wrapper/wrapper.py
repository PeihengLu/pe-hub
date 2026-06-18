"""Wrapper that does not implement the full BasePEModel contract."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from pe_common.model_interface import BasePEModel


class BrokenWrapper(BasePEModel):
    def load_model(self, model_path: str) -> None:
        pass

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    def predict(self, data: pd.DataFrame, batch_size: int = 32) -> List[float]:
        return [0.0] * len(data)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data=None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {}

    def save_model(self, model_path: str) -> None:
        pass

    def save_to_registry(self, dest_dir) -> str:
        return "dummy_state_dict"

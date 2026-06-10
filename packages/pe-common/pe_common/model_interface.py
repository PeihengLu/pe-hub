from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch


class BasePEModel(ABC):
    """Base interface for all Prime Editing prediction models"""
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        from .devices import resolve_device

        self.model_name = model_name
        self.device = device if device is not None else resolve_device()
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model from disk"""
        pass
    
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        """Prepare data for model input"""
        pass
    
    @abstractmethod
    def predict(self, data: Any, batch_size: int = 32) -> List[float]:
        """Make predictions on prepared data"""
        pass
    
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, 
              hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        """Evaluate the model on held-out data using a registered weight set.

        Args:
            test_data: Test data with ground-truth labels.
            weights: Registered weight set ID (see ``list_available_weights()``).
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            'name': self.model_name,
            'device': str(self.device),
            'is_trained': self.is_trained
        }

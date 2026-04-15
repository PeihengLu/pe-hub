import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path

# Ensure vendor models are importable in local development
_vendor_root = resolve_vendor_models_path()
sys.path.insert(0, str(_vendor_root))

from pe_common.model_interface import BasePEModel


class DeepPrimeModelWrapper(BasePEModel):
    """Wrapper for DeepPrime model"""
    
    # Available PE systems and cell types
    SUPPORTED_PE_SYSTEMS = ['PE2', 'PE2max', 'PE4max', 'PE2max-e', 'PE4max-e', 
                           'NRCH_PE2', 'NRCH_PE2max', 'NRCH_PE4max', 'PE2-Off']
    
    SUPPORTED_CELL_TYPES = ['HEK293T', 'A549', 'DLD1', 'HCT116', 'HeLa', 
                           'MDA-MB-231', 'NIH3T3']
    
    def __init__(self, device: Optional[torch.device] = None, 
                 pe_system: str = 'PE2max', 
                 cell_type: str = 'HEK293T'):
        """
        Initialize DeepPrime model wrapper
        
        Args:
            device: PyTorch device
            pe_system: Prime editor system (PE2, PE2max, PE4max, etc.)
            cell_type: Cell type (HEK293T, A549, DLD1, etc.)
        """
        super().__init__('DeepPrime', device)
        
        if pe_system not in self.SUPPORTED_PE_SYSTEMS:
            raise ValueError(
                f"Unsupported PE system: {pe_system}. "
                f"Supported: {self.SUPPORTED_PE_SYSTEMS}"
            )
        
        if cell_type not in self.SUPPORTED_CELL_TYPES:
            raise ValueError(
                f"Unsupported cell type: {cell_type}. "
                f"Supported: {self.SUPPORTED_CELL_TYPES}"
            )
        
        self.pe_system = pe_system
        self.cell_type = cell_type
        self.model_dir = None
        self.model_type = None
        self.models = []
        self.mean = None
        self.std = None
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load pre-trained DeepPrime model
        
        Args:
                        model_path: Optional custom model path. If None, uses default model from repository.
        """
        from deepprime.models.load_model import load_deepprime
        from glob import glob
        
        if model_path:
            # Custom model path
            self.model_dir = os.path.dirname(model_path)
            self.model_type = os.path.basename(model_path)
        else:
            # Use default model from repository
            self.model_dir, self.model_type = load_deepprime(
                self.pe_system, 
                self.cell_type, 
                silent=True
            )
        
        # Load normalization parameters
        mean_path = f'{self.model_dir}/DeepPrime_base/mean.csv'
        std_path = f'{self.model_dir}/DeepPrime_base/std.csv'
        
        self.mean = pd.read_csv(mean_path, header=None, index_col=0).squeeze()
        self.std = pd.read_csv(std_path, header=None, index_col=0).squeeze()
        
        # Load ensemble models
        from deepprime.src.dprime import GeneInteractionModel
        
        model_files = glob(f'{self.model_dir}/{self.model_type}/*.pt')
        
        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {self.model_dir}/{self.model_type}"
            )
        
        self.models = []
        for m_path in model_files:
            model = GeneInteractionModel(hidden_size=128, num_layers=1).to(self.device)
            model.load_state_dict(
                torch.load(m_path, map_location=torch.device(self.device))
            )
            model.eval()
            self.models.append(model)
        
        self.model = self.models  # Store for consistency
        self.is_trained = True
    
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare data in DeepPrime format
        
        Args:
            df: DataFrame with required columns for DeepPrime input
            
        Returns:
            Dictionary with 'g' (gene features) and 'x' (other features) tensors
        """
        from deepprime.src.utils import seq_concat, select_cols
        
        # Extract gene sequence features
        g_features = seq_concat(df)
        
        # Extract and normalize other features
        x_features = select_cols(df)
        x_normalized = (x_features - self.mean) / self.std
        
        # Convert to tensors
        g_tensor = torch.tensor(g_features, dtype=torch.float32, device=self.device)
        x_tensor = torch.tensor(x_normalized.to_numpy(), dtype=torch.float32, device=self.device)
        
        return {'g': g_tensor, 'x': x_tensor}
    
    def predict(self, data: Dict[str, torch.Tensor], batch_size: int = 32) -> List[float]:
        """
        Make predictions using DeepPrime model ensemble
        
        Args:
            data: Dictionary with 'g' and 'x' tensors from prepare_data
            batch_size: Batch size for prediction (not used, kept for API consistency)
            
        Returns:
            List of predicted PE efficiencies
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        g_tensor = data['g']
        x_tensor = data['x']
        
        # Permute gene features for conv2d input
        g_tensor = g_tensor.permute((0, 3, 1, 2))
        
        # Collect predictions from ensemble
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(g_tensor, x_tensor).detach().cpu().numpy()
            preds.append(pred)
        
        # Average ensemble predictions
        preds = np.squeeze(np.array(preds))
        preds = np.mean(preds, axis=0)
        
        # Transform predictions (softplus inverse)
        preds = np.exp(preds) - 1
        
        return preds.tolist() if isinstance(preds, np.ndarray) else [preds]
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train DeepPrime model
        
        Note: Training requires additional implementation details not available
        in the inference-only repository.
        """
        raise NotImplementedError(
            "DeepPrime training interface not yet implemented. "
            "DeepPrime models are typically used as pre-trained models. "
            "Please use the official DeepPrime training pipeline if you need to train custom models."
        )
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate DeepPrime model
        
        Args:
            test_data: DataFrame with input features and 'Efficiency' column
            
        Returns:
            Dictionary with evaluation metrics (Pearson, Spearman)
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from scipy.stats import pearsonr, spearmanr
        
        # Separate features and labels
        if 'Efficiency' in test_data.columns:
            y_true = test_data['Efficiency'].values
            X_test = test_data.drop('Efficiency', axis=1)
        elif 'PE_efficiency' in test_data.columns:
            y_true = test_data['PE_efficiency'].values
            X_test = test_data.drop('PE_efficiency', axis=1)
        else:
            raise ValueError("Test data must contain 'Efficiency' or 'PE_efficiency' column")
        
        # Prepare data and make predictions
        prepared_data = self.prepare_data(X_test)
        y_pred = np.array(self.predict(prepared_data))
        
        # Calculate metrics
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        # Calculate additional metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr),
            'mse': float(mse),
            'mae': float(mae),
            'n_samples': len(y_true)
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save trained DeepPrime model
        
        Note: Typically DeepPrime models are not retrained, but this can save
        the current ensemble if needed.
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save each model in the ensemble
        for idx, model in enumerate(self.models):
            save_path = os.path.join(model_path, f'model_{idx}.pt')
            torch.save(model.state_dict(), save_path)
        
        # Save normalization parameters
        self.mean.to_csv(os.path.join(model_path, 'mean.csv'))
        self.std.to_csv(os.path.join(model_path, 'std.csv'))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'pe_system': self.pe_system,
            'cell_type': self.cell_type,
            'n_models': len(self.models),
            'model_type': self.model_type,
            'supported_pe_systems': self.SUPPORTED_PE_SYSTEMS,
            'supported_cell_types': self.SUPPORTED_CELL_TYPES
        })
        return info

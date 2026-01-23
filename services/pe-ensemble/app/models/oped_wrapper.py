import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path
from pe_common.model_interface import BasePEModel
from pe_common.constants import DEVICE

# Ensure that the OPED model code directory is in sys.path
_vendor_root = resolve_vendor_models_path()
sys.path.insert(0, str(_vendor_root))

class OPEDModelWrapper(BasePEModel):
    """Wrapper for OPED (Order-based Prediction of Editing outcomes and Deletion) model"""
    
    def __init__(self, device: torch.device = torch.device(DEVICE)):
        super().__init__('OPED', device)
        self.model_dir = None
        self.encoder_model = None
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained OPED model
        
        Args:
            model_path: Path to the saved OPED model file (.pt or .pkl)
        """
        from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderModelOrder3
        
        self.model_dir = os.path.dirname(model_path)
        model_file = os.path.basename(model_path)
        
        # Initialize model architecture (these should match the trained model)
        # Default parameters - adjust based on your trained model
        self.encoder_model = TransformerEncoderModelOrder3(
            ntoken=4,
            embedding_size=64,
            hidden_size=[2048, 2048, 2048],
            hidden_size_fully=None,
            output_size=1,
            nhead=8,
            num_encoder_layers=[6, 6, 6],
            dropout=0.1,
            other_size=0
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.encoder_model.load_state_dict(state_dict)
        self.encoder_model.to(self.device)
        self.encoder_model.eval()
        
        self.model = self.encoder_model
        self.is_trained = True
    
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for the OPED model
        
        Args:
            df: DataFrame with standard pegRNA features
            
        Returns:
            DataFrame ready for OPED prediction
        """
        from oped.pegRNA_PredictingCodes.read_data import read_data_of_ClinVar_file
        
        # Convert standardized format to OPED format
        # OPED expects specific column names and data structure
        prepared_data = read_data_of_ClinVar_file(df)
        
        return prepared_data
    
    def predict(self, data: pd.DataFrame, batch_size: int = 1024) -> List[float]:
        """
        Make predictions using OPED model
        
        Args:
            data: Prepared DataFrame from prepare_data()
            batch_size: Batch size for prediction
            
        Returns:
            List of predicted PE efficiencies
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from pegRNA_PredictingCodes.evaluate_model import transformer_predictor_order3
        
        outputs, _ = transformer_predictor_order3(
            transformer=self.encoder_model,
            X_test=data,
            batch_size_test=batch_size,
            device=self.device
        )
        
        return outputs.tolist() if isinstance(outputs, np.ndarray) else outputs
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              hyperparameters: Optional[Dict[str, Any]] = None, freezing: bool = False) -> Dict[str, Any]:
        """
        Train OPED model
        
        Args:
            train_data: Training DataFrame with features and 'Efficiency' label
            val_data: Optional validation DataFrame
            hyperparameters: Training hyperparameters
            freezing: Whether to freeze the representation layers during training
        Returns:
            Dictionary with training results
        """
        from pegRNA_PredictingCodes.train_model import train_and_test_transformer_order3, TransformerEncoderModelOrder3
        
        # Prepare training data
        X_train = self.prepare_data(train_data.drop('Efficiency', axis=1, errors='ignore'))
        y_train = train_data['Efficiency'].values
        
        if val_data is not None:
            X_val = self.prepare_data(val_data.drop('Efficiency', axis=1, errors='ignore'))
            y_val = val_data['Efficiency'].values
        else:
            # Use training data as validation if not provided
            X_val, y_val = X_train, y_train
        
        # Default hyperparameters for OPED transformer
        default_params = {
            'ntoken': 4,
            'embedding_size': 64,
            'hidden_size': [2048, 2048, 2048],
            'hidden_size_fully': None,
            'output_size': 1,
            'nhead': 8,
            'num_encoder_layers': [6, 6, 6],
            'drop_out': 0.1,
            'epoch_num': 100,
            'batch_size': 128,
            'lr': 0.001,
            'weight_decay': 0.0,
            'device': self.device,
            'best_epoch': True,
            'transfer': False,
            'freezing': freezing,
            'other_size': 0
        }
        
        if hyperparameters:
            default_params.update(hyperparameters)
        
        # Train model
        trained_model = train_and_test_transformer_order3(
            X_train=X_train,
            X_test=X_val,
            y_train=y_train,
            y_test=y_val,
            hyperparameters=default_params,
            transformer=None  # Train from scratch
        )
        
        self.encoder_model = trained_model
        self.model = trained_model
        self.is_trained = True
        
        # Evaluate on validation set
        from pegRNA_PredictingCodes.evaluate_model import evaluate_transformer_order3
        
        results, _, _ = evaluate_transformer_order3(
            transformer=self.encoder_model,
            X_train=X_val,
            y_train=y_val,
            batch_size_test=1024,
            device=self.device,
            verbose=True
        )
        
        return {
            'status': 'success',
            'hyperparameters': default_params,
            'val_pearson': results['pearson'][0],
            'val_spearman': results['spearman'][0]
        }
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate OPED model
        
        Args:
            test_data: Test DataFrame with features and 'Efficiency' label
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from pegRNA_PredictingCodes.evaluate_model import evaluate_transformer_order3
        
        # Prepare test data
        X_test = self.prepare_data(test_data.drop('Efficiency', axis=1, errors='ignore'))
        y_test = test_data['Efficiency'].values
        
        # Evaluate
        results, _, _ = evaluate_transformer_order3(
            transformer=self.encoder_model,
            X_train=X_test,
            y_train=y_test,
            batch_size_test=1024,
            device=self.device,
            verbose=True
        )
        
        return {
            'pearson': float(results['pearson'][0]),
            'spearman': float(results['spearman'][0]),
            'n_samples': len(y_test)
        }
    
    def save_model(self, model_path: str) -> None:
        """
        Save trained OPED model
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dict
        torch.save(self.encoder_model.state_dict(), model_path)
        
        print(f"Model saved to {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'model_type': 'Transformer (Order 3)',
            'architecture': 'Encoder-only Transformer with attention',
            'description': 'OPED model for predicting Prime Editing efficiency'
        })
        return info

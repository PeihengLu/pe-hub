import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path

# Add vendor model path
_vendor_root = resolve_vendor_models_path()
sys.path.insert(0, str(_vendor_root))

from pe_common.model_interface import BasePEModel


class PRIDICT2ModelWrapper(BasePEModel):
    """Wrapper for PRIDICT2 model (outcome distribution prediction)"""
    
    SUPPORTED_CELL_TYPES = ['HEK', 'K562', 'HEKschwank', 'HEKhyongbum']
    
    SUPPORTED_MODEL_NAMES = [
        'base_90k',
        'base_390k',
        'base_23k',
        'base_90k_decinit_HEKschwank_FT',
        'base_390k_decinit_HEKhyongbum_FT',
        'base_390k_decinit_HEKschwank_FT'
    ]
    
    def __init__(self, device: Optional[torch.device] = None, 
                 wsize: int = 20,
                 model_name: str = 'base_390k'):
        """
        Initialize PRIDICT2 model wrapper
        
        Args:
            device: PyTorch device
            wsize: Window size for sequence processing
            model_name: Pre-trained model name to use
        """
        super().__init__('PRIDICT2', device)
        self.wsize = wsize
        self.model_name_str = model_name
        
        if model_name not in self.SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"Unsupported model name: {model_name}. "
                f"Supported: {self.SUPPORTED_MODEL_NAMES}"
            )
        
        from pridict.pridictv2.predict_outcomedistrib import PRIEML_Model
        
        self.prieml_model = PRIEML_Model(
            device=device or torch.device('cpu'),
            wsize=wsize,
            normalize='max',
            fdtype=torch.float32
        )
        self.model_components = None
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained PRIDICT2 model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_components = self.prieml_model.build_retrieve_models(model_path)
        self.model = self.model_components
        self.is_trained = True
    
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Prepare data in PRIDICT2 format
        
        Args:
            df: DataFrame with standard pegRNA features
            **kwargs: Additional arguments
                - y_ref: List of target columns (default: ['averageedited', 'averageunedited', 'averageindel'])
                - batch_size: Batch size for DataLoader (default: 500)
                - cell_types: List of cell types for each sample
                
        Returns:
            DataLoader ready for PRIDICT2 prediction
        """
        y_ref = kwargs.get('y_ref', ['averageedited', 'averageunedited', 'averageindel'])
        batch_size = kwargs.get('batch_size', 500)
        cell_types = kwargs.get('cell_types', None)
        
        # Prepare data using PRIDICT2's preprocessing pipeline
        dloader = self.prieml_model.prepare_data(
            df=df,
            model_name=self.model_name_str,
            cell_types=cell_types,
            y_ref=y_ref,
            batch_size=batch_size
        )
        
        return dloader
    
    def predict(self, data: Any, batch_size: int = 500) -> List[List[float]]:
        """
        Make predictions using PRIDICT2 model
        
        Args:
            data: DataLoader from prepare_data()
            batch_size: Batch size (not used, kept for API consistency)
            
        Returns:
            List of predictions [edited, unedited, indel] for each sample
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Make predictions for all three outcomes
        pred_df = self.prieml_model.predict_from_dloader(
            dloader=data,
            model_dir=None,  # model already loaded
            y_ref=['averageedited', 'averageunedited', 'averageindel']
        )
        
        # Return all three outcomes as a list of lists
        predictions = pred_df[[
            'pred_averageedited',
            'pred_averageunedited', 
            'pred_averageindel'
        ]].values.tolist()
        
        return predictions
    
    def predict_single_outcome(self, data: Any, outcome: str = 'averageedited') -> List[float]:
        """
        Make predictions for a single outcome
        
        Args:
            data: DataLoader from prepare_data()
            outcome: Which outcome to predict ('averageedited', 'averageunedited', or 'averageindel')
            
        Returns:
            List of predicted values for the specified outcome
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        valid_outcomes = ['averageedited', 'averageunedited', 'averageindel']
        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
        
        pred_df = self.prieml_model.predict_from_dloader(
            dloader=data,
            model_dir=None,
            y_ref=[outcome]
        )
        
        return pred_df[f'pred_{outcome}'].tolist()
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train PRIDICT2 model
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            hyperparameters: Training hyperparameters
            
        Returns:
            Dictionary with training results
        """
        from pridict.pridictv2.run_workflow import build_config_map, run_cont_pe_RNN_distribution
        
        # This requires setting up the full training pipeline
        # with proper data loaders, model initialization, etc.
        # For now, raise NotImplementedError with guidance
        
        raise NotImplementedError(
            "PRIDICT2 training requires complex setup. "
            "To train PRIDICT2:\n"
            "1. Prepare data in PRIDICT2 format with all required columns\n"
            "2. Use pridict.pridictv2.run_workflow module directly\n"
            "3. Configure hyperparameters using RNNHyperparamConfig\n"
            "4. Call run_cont_pe_RNN_distribution() with proper data partition\n"
            "See vendor/models/pridict2/notebooks/ for training examples."
        )
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate PRIDICT2 model on all three outcomes
        
        Args:
            test_data: Test DataFrame with true labels
            
        Returns:
            Dictionary with evaluation metrics for each outcome
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from scipy.stats import pearsonr, spearmanr
        
        # Prepare data
        dloader = self.prepare_data(
            test_data,
            y_ref=['averageedited', 'averageunedited', 'averageindel']
        )
        
        # Make predictions
        pred_df = self.prieml_model.predict_from_dloader(
            dloader=dloader,
            model_dir=None,
            y_ref=['averageedited', 'averageunedited', 'averageindel']
        )
        
        results = {}
        
        # Evaluate each outcome
        for outcome in ['averageedited', 'averageunedited', 'averageindel']:
            y_true = pred_df[f'true_{outcome}'].values
            y_pred = pred_df[f'pred_{outcome}'].values
            
            # Remove NaN values if present
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                pearson_corr, _ = pearsonr(y_true_clean, y_pred_clean)
                spearman_corr, _ = spearmanr(y_true_clean, y_pred_clean)
                mse = np.mean((y_true_clean - y_pred_clean) ** 2)
                mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                
                results[f'{outcome}_pearson'] = float(pearson_corr)
                results[f'{outcome}_spearman'] = float(spearman_corr)
                results[f'{outcome}_mse'] = float(mse)
                results[f'{outcome}_mae'] = float(mae)
            else:
                results[f'{outcome}_pearson'] = np.nan
                results[f'{outcome}_spearman'] = np.nan
                results[f'{outcome}_mse'] = np.nan
                results[f'{outcome}_mae'] = np.nan
        
        results['n_samples'] = len(y_true)
        
        return results
    
    def save_model(self, model_path: str) -> None:
        """
        Save trained PRIDICT2 model
        
        Args:
            model_path: Directory path to save the model components
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        import os
        
        os.makedirs(model_path, exist_ok=True)
        
        # PRIDICT2 models are complex with multiple components
        # This would require saving all model components separately
        raise NotImplementedError(
            "PRIDICT2 model saving not yet fully implemented. "
            "Models typically consist of multiple PyTorch modules that need to be saved separately."
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name_str,
            'wsize': self.wsize,
            'supported_cell_types': self.SUPPORTED_CELL_TYPES,
            'supported_model_names': self.SUPPORTED_MODEL_NAMES,
            'available_cell_types': self.prieml_model.get_celltypes(self.model_name_str),
            'outcomes': ['averageedited', 'averageunedited', 'averageindel'],
            'description': 'PRIDICT2 model for predicting outcome distribution (edited/unedited/indel)'
        })
        return info
    
    @staticmethod
    def get_supported_models() -> Dict[str, List[str]]:
        """Get mapping of model names to supported cell types"""
        return {
            'base_90k': ['HEK'],
            'base_390k': ['HEKschwank', 'HEKhyongbum'],
            'base_23k': ['HEK', 'K562'],
            'base_90k_decinit_HEKschwank_FT': ['HEK', 'K562'],
            'base_390k_decinit_HEKhyongbum_FT': ['HEK', 'K562'],
            'base_390k_decinit_HEKschwank_FT': ['HEK', 'K562']
        }

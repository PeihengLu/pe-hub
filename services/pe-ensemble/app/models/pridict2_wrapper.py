import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path
from pe_common.preprocessing import (
    ensure_schema,
    standardized_to_pridict_dataframe,
    STANDARDIZED_BASE_COLUMNS,
)

# Add vendor model paths required by PRIDICT2 imports.
_vendor_root = resolve_vendor_models_path()
_pridict2_root = resolve_vendor_models_path("pridict2")
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))
if str(_pridict2_root) not in sys.path:
    sys.path.insert(0, str(_pridict2_root))

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
    STANDARDIZED_REQUIRED_COLUMNS = STANDARDIZED_BASE_COLUMNS
    
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
        
        from pridict2.pridict.pridictv2.predict_outcomedistrib import PRIEML_Model
        
        self.prieml_model = PRIEML_Model(
            device=device or torch.device('cpu'),
            wsize=wsize,
            normalize='max',
            fdtype=torch.float32
        )
        self.model_components = None
        self.loaded_model_dir: Optional[str] = None

    def _default_outcomes(self) -> List[str]:
        # Original PRIDICT1 base models were mostly used for intended edits.
        if self.model_name_str in {"base_90k", "base_390k"}:
            return ["averageedited"]
        return ["averageedited", "averageunedited", "averageindel"]

    def _to_pridict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return ensure_schema(
            df,
            native_required={
                "seq_id",
                "wide_initial_target",
                "wide_mutated_target",
                "deepeditposition",
                "deepeditposition_lst",
                "Correction_Type",
                "Correction_Length",
                "protospacerlocation_only_initial",
                "PBSlocation",
                "RT_initial_location",
                "RT_mutated_location",
            },
            converters={"standardized_to_pridict": standardized_to_pridict_dataframe},
        )

    def _predict_from_loaded_or_current_model(
        self, dloader: Any, y_ref: List[str]
    ) -> pd.DataFrame:
        if self.model_components is not None:
            # PRIDICT2 wrapper exposes a "loaded-models" prediction entrypoint.
            return self.prieml_model.predict_from_dloader_using_loaded_models(
                dloader=dloader,
                models=self.model_components,
                y_ref=y_ref,
            )
        if self.loaded_model_dir:
            return self.prieml_model.predict_from_dloader(
                dloader=dloader,
                model_dir=self.loaded_model_dir,
                y_ref=y_ref,
            )
        raise ValueError("Model not loaded. Call load_model() first.")
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained PRIDICT2 model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.loaded_model_dir = model_path
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
        df = self._to_pridict_dataframe(df)
        y_ref = kwargs.get('y_ref', self._default_outcomes())
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
    
    def predict(self, data: Any, batch_size: int = 500) -> List[float]:
        """
        Make predictions using PRIDICT model.
        
        Args:
            data: DataLoader from prepare_data()
            batch_size: Batch size (not used, kept for API consistency)
            
        Returns:
            List of intended edit predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        pred_df = self._predict_from_loaded_or_current_model(
            dloader=data,
            y_ref=["averageedited"],
        )
        return pred_df["pred_averageedited"].astype(float).tolist()

    def predict_distribution(
        self,
        data: Any,
        outcomes: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """Predict one or multiple PRIDICT outcomes in batch."""
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        y_ref = outcomes or self._default_outcomes()
        pred_df = self._predict_from_loaded_or_current_model(dloader=data, y_ref=y_ref)
        return pred_df[[f"pred_{outcome}" for outcome in y_ref]].values.tolist()
    
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
        
        pred_df = self._predict_from_loaded_or_current_model(dloader=data, y_ref=[outcome])
        
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
        from torch import nn
        from pridict2.pridict.pridictv2.run_workflow import build_config_map, train_val_run

        hyperparameters = hyperparameters or {}
        y_ref = list(hyperparameters.get("y_ref", self._default_outcomes()) or self._default_outcomes())
        train_df = self._to_pridict_dataframe(train_data)
        if val_data is not None:
            val_df = self._to_pridict_dataframe(val_data)
        else:
            val_df = train_df.sample(frac=0.2, random_state=int(hyperparameters.get("random_state", 42)))
            train_df = train_df.drop(index=val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)

        norm_cols_tr, proc_tr, init_tr, n_init_tr, mut_tr, n_mut_tr = self.prieml_model._process_df(train_df)
        dtensor_train = self.prieml_model._construct_datatensor(
            norm_cols_tr, proc_tr, init_tr, n_init_tr, mut_tr, n_mut_tr, y_ref=y_ref
        )
        norm_cols_val, proc_val, init_val, n_init_val, mut_val, n_mut_val = self.prieml_model._process_df(val_df)
        dtensor_val = self.prieml_model._construct_datatensor(
            norm_cols_val, proc_val, init_val, n_init_val, mut_val, n_mut_val, y_ref=y_ref
        )

        data_partitions = {0: {"train": dtensor_train, "validation": dtensor_val}}
        run_gpu_map = {0: int(hyperparameters.get("gpu_index", 0))}
        output_dir = str(hyperparameters.get("output_dir", "artifacts/pridict2_train"))

        batch_size = int(hyperparameters.get("batch_size", 128))
        num_epochs = int(hyperparameters.get("num_epochs", 20))
        trf_tup = hyperparameters.get(
            "trf_tup",
            (64, 64, 1, True, 0.1, nn.GRU, nn.ReLU(), 1e-4, batch_size, num_epochs),
        )
        experiment_options = {
            "experiment_desc": str(hyperparameters.get("experiment_desc", "pe_ensemble_pridict_train")),
            "model_name": "PE_RNN_distribution",
            "annot_embed": int(hyperparameters.get("annot_embed", 8)),
            "assemb_opt": str(hyperparameters.get("assemb_opt", "add")),
            "seqlevel_featdim": int(hyperparameters.get("seqlevel_featdim", len(norm_cols_tr or []))),
            "num_outcomes": int(hyperparameters.get("num_outcomes", len(y_ref))),
        }
        config_map = build_config_map(trf_tup, experiment_options, loss_func=str(hyperparameters.get("loss_func", "KLDloss")))
        train_val_run(
            datatensor_partitions=data_partitions,
            config_map=config_map,
            train_val_dir=output_dir,
            run_gpu_map=run_gpu_map,
            num_epochs=num_epochs,
        )

        run_dir = f"{output_dir}/train_val/run_0"
        self.load_model(run_dir)
        return {
            "status": "success",
            "output_dir": output_dir,
            "model_dir": run_dir,
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "outcomes": y_ref,
        }
    
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
        
        test_df = self._to_pridict_dataframe(test_data)
        outcomes = [o for o in self._default_outcomes() if o in test_df.columns]
        if not outcomes:
            outcomes = ["averageedited"]

        # Prepare data
        dloader = self.prepare_data(
            test_df,
            y_ref=outcomes
        )
        
        # Make predictions
        pred_df = self._predict_from_loaded_or_current_model(dloader=dloader, y_ref=outcomes)
        
        results = {}
        n_samples = 0
        
        # Evaluate each outcome
        for outcome in outcomes:
            y_true = pred_df[f'true_{outcome}'].values
            y_pred = pred_df[f'pred_{outcome}'].values
            n_samples = len(y_true)
            
            # Remove NaN values if present
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                pearson_res = pearsonr(y_true_clean, y_pred_clean)
                spearman_res = spearmanr(y_true_clean, y_pred_clean)
                pearson_corr = float(np.asarray(pearson_res).reshape(-1)[0])
                spearman_corr = float(np.asarray(spearman_res).reshape(-1)[0])
                mse = np.mean((y_true_clean - y_pred_clean) ** 2)
                mae = np.mean(np.abs(y_true_clean - y_pred_clean))
                
                results[f'{outcome}_pearson'] = pearson_corr
                results[f'{outcome}_spearman'] = spearman_corr
                results[f'{outcome}_mse'] = float(mse)
                results[f'{outcome}_mae'] = float(mae)
            else:
                results[f'{outcome}_pearson'] = np.nan
                results[f'{outcome}_spearman'] = np.nan
                results[f'{outcome}_mse'] = np.nan
                results[f'{outcome}_mae'] = np.nan
        
        results['n_samples'] = n_samples
        
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
            'outcomes': self._default_outcomes(),
            'supports_standardized_input': True,
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

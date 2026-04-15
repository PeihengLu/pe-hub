# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
import sys
import os
from typing import List, Dict, Any, Optional, cast
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path

# Ensure vendor models are importable in local development
_vendor_root = resolve_vendor_models_path()
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))

from pe_common.model_interface import BasePEModel
from pe_common.preprocessing import ensure_schema, standardized_to_deepprime_features


class DeepPrimeModelWrapper(BasePEModel):
    """Wrapper for DeepPrime model"""
    
    # Available PE systems and cell types
    SUPPORTED_PE_SYSTEMS = ['PE2', 'PE2max', 'PE4max', 'PE2max-e', 'PE4max-e', 
                           'NRCH_PE2', 'NRCH_PE2max', 'NRCH_PE4max', 'PE2-Off']
    
    SUPPORTED_CELL_TYPES = ['HEK293T', 'A549', 'DLD1', 'HCT116', 'HeLa', 
                           'MDA-MB-231', 'NIH3T3']
    DEEPPRIME_REQUIRED_COLUMNS = {
        'WT74_On', 'Edited74_On', 'PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos',
        'Edit_len', 'RHA_len', 'type_sub', 'type_ins', 'type_del', 'Tm1', 'Tm2',
        'Tm2new', 'Tm3', 'Tm4', 'TmD', 'nGCcnt1', 'nGCcnt2', 'nGCcnt3',
        'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score',
    }
    
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
        self.mean: Optional[pd.Series] = None
        self.std: Optional[pd.Series] = None
        self._last_training_history: List[Dict[str, float]] = []

    def _to_deepprime_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return ensure_schema(
            df,
            native_required=self.DEEPPRIME_REQUIRED_COLUMNS,
            converters={"standardized_to_deepprime": standardized_to_deepprime_features},
        )
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load pre-trained DeepPrime model
        
        Args:
                        model_path: Optional custom model path. If None, uses default model from repository.
        """
        from glob import glob
        from deepprime.models.load_model import load_deepprime
        
        if model_path:
            # Custom model path
            model_path = str(model_path)
            if model_path.endswith(".pt"):
                self.model_dir = os.path.dirname(os.path.dirname(model_path))
                self.model_type = os.path.basename(os.path.dirname(model_path))
            else:
                self.model_dir = os.path.dirname(model_path)
                self.model_type = os.path.basename(model_path.rstrip("/"))
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
        
        mean_obj = pd.read_csv(mean_path, header=None, index_col=0).squeeze()
        std_obj = pd.read_csv(std_path, header=None, index_col=0).squeeze()
        self.mean = cast(pd.Series, mean_obj if isinstance(mean_obj, pd.Series) else pd.Series(dtype=float))
        self.std = cast(pd.Series, std_obj if isinstance(std_obj, pd.Series) else pd.Series(dtype=float))
        
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
            df: DataFrame in DeepPrime feature schema or standardized schema
            
        Returns:
            Dictionary with 'g' (gene features) and 'x' (other features) tensors
        """
        from deepprime.src.utils import seq_concat, select_cols

        feature_df = self._to_deepprime_feature_df(df)
        # Extract gene sequence features
        g_features = seq_concat(feature_df)

        # Extract and normalize other features
        x_features = select_cols(feature_df)
        if self.mean is not None and self.std is not None:
            mean_series = self.mean.reindex(x_features.columns).fillna(0.0)
            std_series = self.std.reindex(x_features.columns).replace(0, 1.0).fillna(1.0)
            x_features = x_features.fillna(mean_series)
            x_processed = (x_features - mean_series) / std_series
        else:
            x_processed = x_features.fillna(0.0)

        # Convert to tensors
        g_tensor = torch.tensor(g_features, dtype=torch.float32, device=self.device)
        x_tensor = torch.tensor(x_processed.to_numpy(), dtype=torch.float32, device=self.device)
        
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
        Fine-tune loaded DeepPrime model(s) on prepared data.

        This is a lightweight fine-tuning path (not the original DeepPrime
        training pipeline). It supports standardized schema by converting into
        DeepPrime feature columns before optimization.
        """
        hyperparameters = hyperparameters or {}
        epochs = int(hyperparameters.get("epochs", 5))
        batch_size = int(hyperparameters.get("batch_size", 128))
        lr = float(hyperparameters.get("lr", 1e-4))
        weight_decay = float(hyperparameters.get("weight_decay", 0.0))
        train_ensemble = bool(hyperparameters.get("train_ensemble", False))
        load_pretrained = bool(hyperparameters.get("load_pretrained", True))

        train_feature_df = self._to_deepprime_feature_df(train_data)
        if "editing_efficiency" in train_data.columns:
            y_train = pd.Series(
                pd.to_numeric(train_data["editing_efficiency"], errors="coerce"),
                index=train_data.index,
            ).fillna(0.0).to_numpy(dtype=np.float32)
        elif "Efficiency" in train_feature_df.columns:
            y_train = pd.Series(
                pd.to_numeric(train_feature_df["Efficiency"], errors="coerce"),
                index=train_feature_df.index,
            ).fillna(0.0).to_numpy(dtype=np.float32)
        else:
            raise ValueError("Training data must include 'editing_efficiency' or 'Efficiency' column.")

        if val_data is not None:
            val_feature_df = self._to_deepprime_feature_df(val_data)
            if "editing_efficiency" in val_data.columns:
                y_val = pd.Series(
                    pd.to_numeric(val_data["editing_efficiency"], errors="coerce"),
                    index=val_data.index,
                ).fillna(0.0).to_numpy(dtype=np.float32)
            elif "Efficiency" in val_feature_df.columns:
                y_val = pd.Series(
                    pd.to_numeric(val_feature_df["Efficiency"], errors="coerce"),
                    index=val_feature_df.index,
                ).fillna(0.0).to_numpy(dtype=np.float32)
            else:
                raise ValueError("Validation data must include 'editing_efficiency' or 'Efficiency' column.")
        else:
            val_feature_df = None
            y_val = None

        if not self.is_trained:
            if load_pretrained:
                self.load_model()
            else:
                from deepprime.src.dprime import GeneInteractionModel
                self.models = [GeneInteractionModel(hidden_size=128, num_layers=1).to(self.device)]
                self.model = self.models
                self.is_trained = True

        train_inputs = self.prepare_data(train_feature_df)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).reshape(-1, 1)
        if y_val is not None and val_feature_df is not None:
            val_inputs = self.prepare_data(val_feature_df)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device).reshape(-1, 1)
        else:
            val_inputs = None
            y_val_t = None

        models_to_train = self.models if train_ensemble else self.models[:1]
        history: List[Dict[str, float]] = []
        for model_idx, model in enumerate(models_to_train):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.MSELoss()

            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(y_train_t.shape[0], device=self.device)
                epoch_losses: List[float] = []
                for start in range(0, y_train_t.shape[0], batch_size):
                    idx = perm[start:start + batch_size]
                    g_batch = train_inputs["g"][idx].permute((0, 3, 1, 2))
                    x_batch = train_inputs["x"][idx]
                    y_batch = y_train_t[idx]

                    optimizer.zero_grad()
                    pred = model(g_batch, x_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu()))

                train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                log_entry: Dict[str, float] = {
                    "model_index": float(model_idx),
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                }
                if val_inputs is not None and y_val_t is not None:
                    model.eval()
                    with torch.no_grad():
                        pred_val = model(val_inputs["g"].permute((0, 3, 1, 2)), val_inputs["x"])
                        val_loss = float(criterion(pred_val, y_val_t).detach().cpu())
                    log_entry["val_loss"] = val_loss
                history.append(log_entry)

        self._last_training_history = history
        self.model = self.models
        self.is_trained = True
        final = history[-1] if history else {}
        return {
            "status": "success",
            "n_models_trained": len(models_to_train),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "final_train_loss": final.get("train_loss"),
            "final_val_loss": final.get("val_loss"),
        }
    
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
        pearson_res = pearsonr(y_true, y_pred)
        spearman_res = spearmanr(y_true, y_pred)
        pearson_corr = float(pearson_res[0])
        spearman_corr = float(spearman_res[0])
        
        # Calculate additional metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
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
        if self.mean is not None:
            self.mean.to_csv(os.path.join(model_path, 'mean.csv'))
        if self.std is not None:
            self.std.to_csv(os.path.join(model_path, 'std.csv'))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'pe_system': self.pe_system,
            'cell_type': self.cell_type,
            'n_models': len(self.models),
            'model_type': self.model_type,
            'supports_standardized_input': True,
            'supported_pe_systems': self.SUPPORTED_PE_SYSTEMS,
            'supported_cell_types': self.SUPPORTED_CELL_TYPES
        })
        return info

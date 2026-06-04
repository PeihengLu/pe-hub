# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
import sys
import os
from typing import List, Dict, Any, Optional, Tuple, cast
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

from .vendor_path import resolve_vendor_models_path

# Ensure vendor models are importable in local development
_vendor_root = resolve_vendor_models_path()
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))

from pe_common.model_interface import BasePEModel
from pe_common.training import (
    build_lr_scheduler,
    build_group_kfold_indices,
    fit_lightning_module,
    LightningTrainerConfig,
)


class _DeepPrimeTensorDataset(Dataset):
    def __init__(self, g: torch.Tensor, x: torch.Tensor, y: np.ndarray) -> None:
        self.g = g.detach().cpu()
        self.x = x.detach().cpu()
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.g[idx], self.x[idx], self.y[idx]


class _DeepPrimeLightningRegressor(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, train_hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.hparams_map = dict(train_hparams)
        self.criterion = torch.nn.MSELoss()

    def _predict_batch(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model(g.permute((0, 3, 1, 2)), x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        g, x, y = batch
        pred = self._predict_batch(g, x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        g, x, y = batch
        pred = self._predict_batch(g, x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.hparams_map.get("lr", 1e-4)),
            weight_decay=float(self.hparams_map.get("weight_decay", 0.0)),
        )
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_name=str(self.hparams_map.get("scheduler", "none")),
            scheduler_kwargs=cast(Optional[Dict[str, Any]], self.hparams_map.get("scheduler_kwargs")),
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


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
        """Validate that ``df`` is already in DeepPrime's native feature schema.

        Standardized -> DeepPrime conversion is owned by the PE-DB service; fetch
        model-format data from ``GET /api/filter?...&format=deepprime`` rather than
        passing standardized rows here.
        """
        if self.DEEPPRIME_REQUIRED_COLUMNS.issubset(df.columns):
            return df.copy()
        missing = sorted(self.DEEPPRIME_REQUIRED_COLUMNS.difference(df.columns))
        raise ValueError(
            "DeepPrime expects native feature columns; missing: "
            f"{missing}. Fetch model-format data from PE-DB "
            "(GET /api/filter?...&format=deepprime)."
        )

    @staticmethod
    def _extract_targets(source_df: pd.DataFrame, feature_df: pd.DataFrame, split_name: str) -> np.ndarray:
        if "editing_efficiency" in source_df.columns:
            series = pd.Series(pd.to_numeric(source_df["editing_efficiency"], errors="coerce"), index=source_df.index)
            return series.fillna(0.0).to_numpy(dtype=np.float32)
        if "Efficiency" in feature_df.columns:
            series = pd.Series(pd.to_numeric(feature_df["Efficiency"], errors="coerce"), index=feature_df.index)
            return series.fillna(0.0).to_numpy(dtype=np.float32)
        raise ValueError(
            f"{split_name} data must include 'editing_efficiency' or 'Efficiency' column."
        )

    @staticmethod
    def _split_train_validation(
        train_source_df: pd.DataFrame,
        train_feature_df: pd.DataFrame,
        y_train: np.ndarray,
        *,
        val_fraction: float,
        random_state: int,
        group_col: str,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        n_rows = len(train_feature_df)
        if n_rows < 2:
            raise ValueError("Need at least 2 rows to train DeepPrime model.")

        val_fraction = max(0.05, min(0.5, float(val_fraction)))
        n_val = max(1, int(np.ceil(n_rows * val_fraction)))
        n_splits = max(2, int(round(1.0 / val_fraction)))
        n_splits = min(n_splits, n_rows)

        folds = build_group_kfold_indices(
            train_source_df.reset_index(drop=True),
            n_splits=n_splits,
            group_col=group_col,
            random_state=random_state,
        )
        if folds:
            train_idx, val_idx = folds[0]
        else:
            rng = np.random.default_rng(random_state)
            indices = np.arange(n_rows)
            rng.shuffle(indices)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

        if len(train_idx) == 0:
            train_idx = np.asarray([int(val_idx[0])], dtype=int)
            val_idx = np.asarray([int(i) for i in range(n_rows) if i != int(train_idx[0])], dtype=int)

        tr_df = train_feature_df.iloc[train_idx].reset_index(drop=True)
        va_df = train_feature_df.iloc[val_idx].reset_index(drop=True)
        tr_y = y_train[train_idx]
        va_y = y_train[val_idx]
        return tr_df, tr_y, va_df, va_y
    
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
    
    @staticmethod
    def list_available_weights() -> List[str]:
        """List the names of pre-trained DeepPrime weight sets bundled in vendor/models.

        Each name corresponds to a model variant directory (for example
        'DeepPrime_base' or 'DP_variant_293T_PE4max_Opti_220728') containing the
        ensemble ``.pt`` checkpoints. Pass one of these names to
        :meth:`evaluate` (``weights=...``) or :meth:`load_weights_by_name`.
        """
        from glob import glob

        models_root = resolve_vendor_models_path("deepprime", "models", "DeepPrime")
        names: List[str] = []
        for entry in sorted(models_root.iterdir()):
            if entry.is_dir() and glob(str(entry / "*.pt")):
                names.append(entry.name)
        return names

    def load_weights_by_name(self, name: str) -> None:
        """Load a named pre-trained DeepPrime weight set.

        Args:
            name: A weight set name from :meth:`list_available_weights` (the
                model variant directory name).
        """
        models_root = resolve_vendor_models_path("deepprime", "models", "DeepPrime")
        target = models_root / name
        if not target.is_dir():
            raise ValueError(
                f"Unknown DeepPrime weights '{name}'. "
                f"Available: {self.list_available_weights()}"
            )
        self.load_model(str(target))

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prepare data in DeepPrime format
        
        Args:
            df: DataFrame in DeepPrime's native feature schema (fetch from
                PE-DB: GET /api/filter?...&format=deepprime)
            
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
        training pipeline). Inputs must already be in DeepPrime's native feature
        schema (fetch model-format data from PE-DB: /api/filter?...&format=deepprime).
        """
        hyperparameters = hyperparameters or {}
        epochs = int(hyperparameters.get("epochs", 5))
        batch_size = int(hyperparameters.get("batch_size", 128))
        lr = float(hyperparameters.get("lr", 1e-4))
        weight_decay = float(hyperparameters.get("weight_decay", 0.0))
        train_ensemble = bool(hyperparameters.get("train_ensemble", False))
        load_pretrained = bool(hyperparameters.get("load_pretrained", True))
        grad_clip = float(hyperparameters.get("grad_clip", 1.0))
        scheduler_name = hyperparameters.get("scheduler", "none")
        scheduler_kwargs = hyperparameters.get("scheduler_kwargs", None)
        early_stopping_patience = int(hyperparameters.get("early_stopping_patience", 10))
        early_stopping_min_delta = float(hyperparameters.get("early_stopping_min_delta", 0.0))
        reshuffle_each_epoch = bool(hyperparameters.get("reshuffle_each_epoch", True))
        val_fraction = float(hyperparameters.get("val_fraction", 0.1))
        val_random_state = int(hyperparameters.get("val_random_state", 42))
        split_group_col = str(hyperparameters.get("split_group_col", "group_id"))

        train_feature_df = self._to_deepprime_feature_df(train_data)
        y_train = self._extract_targets(train_data, train_feature_df, "Training")

        if val_data is not None:
            val_feature_df = self._to_deepprime_feature_df(val_data)
            y_val = self._extract_targets(val_data, val_feature_df, "Validation")
        else:
            train_feature_df, y_train, val_feature_df, y_val = self._split_train_validation(
                train_data,
                train_feature_df,
                y_train,
                val_fraction=val_fraction,
                random_state=val_random_state,
                group_col=split_group_col,
            )

        if not self.is_trained:
            if load_pretrained:
                self.load_model()
            else:
                from deepprime.src.dprime import GeneInteractionModel

                self.models = [GeneInteractionModel(hidden_size=128, num_layers=1).to(self.device)]
                self.model = self.models
                self.is_trained = True

        train_inputs = self.prepare_data(train_feature_df)
        val_inputs = self.prepare_data(val_feature_df)
        train_dataset = _DeepPrimeTensorDataset(train_inputs["g"], train_inputs["x"], y_train)
        val_dataset = _DeepPrimeTensorDataset(val_inputs["g"], val_inputs["x"], y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=reshuffle_each_epoch,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        models_to_train = self.models if train_ensemble else self.models[:1]
        history: List[Dict[str, float]] = []
        model_summaries: List[Dict[str, float]] = []

        for model_idx, model in enumerate(models_to_train):
            lightning_hparams = dict(hyperparameters)
            lightning_hparams["lr"] = lr
            lightning_hparams["weight_decay"] = weight_decay
            lightning_hparams["scheduler"] = scheduler_name
            lightning_hparams["scheduler_kwargs"] = scheduler_kwargs
            lightning_module = _DeepPrimeLightningRegressor(model, lightning_hparams)
            metrics = fit_lightning_module(
                lightning_module,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                config=LightningTrainerConfig(
                    max_epochs=epochs,
                    grad_clip=grad_clip,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    enable_progress_bar=bool(hyperparameters.get("progress_bar", False)),
                    log_every_n_steps=int(hyperparameters.get("log_every_n_steps", 25)),
                ),
            )
            for row in cast(List[Dict[str, float]], metrics["history"]):
                history.append({"model_index": float(model_idx), **row})
            model_summaries.append(
                {
                    "model_index": float(model_idx),
                    "best_epoch": float(metrics["best_epoch"]),
                    "best_val_loss": float(metrics["best_val_loss"]),
                    "n_epochs_ran": float(metrics["n_epochs_ran"]),
                }
            )

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
            "model_summaries": model_summaries,
        }
    
    def evaluate(self, test_data: pd.DataFrame, weights: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate DeepPrime model

        Args:
            test_data: DataFrame with input features and 'Efficiency' column
            weights: Optional name of a bundled pre-trained weight set to load
                before evaluating (see :meth:`list_available_weights`). When
                ``None``, the currently trained/loaded model is used.

        Returns:
            Dictionary with evaluation metrics (Pearson, Spearman)
        """
        if weights is not None:
            self.load_weights_by_name(weights)

        if not self.is_trained:
            raise ValueError(
                "Model not loaded. Pass `weights=<name>` (see list_available_weights()), "
                "or call load_model()/train() first."
            )
        
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

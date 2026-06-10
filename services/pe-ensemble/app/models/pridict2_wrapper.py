import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path
from . import weights_registry
from pe_common.training import pearson_spearman
from pe_common.splits import (
    has_assigned_cv_folds,
    iter_assigned_cv_folds,
    resolve_train_val_from_splits,
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

    # Native PRIDICT/PRIDICT2 input columns (output of PE-DB's pridict converter).
    PRIDICT_REQUIRED_COLUMNS = {
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
    }
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        wsize: int = 20,
        model_name: Optional[str] = None,
    ):
        """
        Initialize PRIDICT2 model wrapper.

        Args:
            device: PyTorch device
            wsize: Window size for sequence processing
            model_name: Optional legacy vendor base-model name used only when
                preparing data without loaded weights (prefer explicit weights).
        """
        super().__init__('PRIDICT2', device)
        self.wsize = wsize
        self.model_name_str = model_name

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
        if self.model_name_str in {None, "base_90k", "base_390k"}:
            return ["averageedited"]
        return ["averageedited", "averageunedited", "averageindel"]

    def _cell_types_from_loaded_config(self) -> List[str]:
        """Return prediction-head cell types from the loaded weight run config."""
        if not self.loaded_model_dir:
            raise ValueError("PRIDICT2 weights are not loaded.")
        import os

        mconfig_dir = os.path.join(self.loaded_model_dir, "config")
        _, options = self.prieml_model._load_model_config(mconfig_dir)
        datasets = options.get("datasets_name") or []
        return ["".join(str(name).split("_")) for name in datasets]

    def _to_pridict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` is already in PRIDICT's native schema.

        Standardized -> PRIDICT conversion is owned by the PE-DB service; fetch
        model-format data from ``GET /api/filter?...&format=pridict2`` rather than
        passing standardized rows here.
        """
        if self.PRIDICT_REQUIRED_COLUMNS.issubset(df.columns):
            return df.copy()
        missing = sorted(self.PRIDICT_REQUIRED_COLUMNS.difference(df.columns))
        raise ValueError(
            "PRIDICT2 expects native input columns; missing: "
            f"{missing}. Fetch model-format data from PE-DB "
            "(GET /api/filter?...&format=pridict2)."
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
    
    @staticmethod
    def list_available_weights() -> List[str]:
        """List registered PRIDICT2 weight set IDs."""
        return weights_registry.list_weight_ids("pridict2")

    def _resolve_weights_dir(self, name: str) -> Path:
        """Resolve a weight set ID (or directory path) to a run directory."""
        candidate = Path(name).expanduser()
        if candidate.is_dir() and (candidate / "model_statedict").is_dir():
            return candidate

        registry_id = name.replace("/", "__")
        try:
            return weights_registry.resolve_dir("pridict2", registry_id)
        except ValueError:
            pass

        if "__" in name:
            parts = name.split("__")
            if len(parts) == 3:
                trained_root = resolve_vendor_models_path("pridict2", "trained_models")
                legacy = trained_root / parts[0] / parts[1] / "train_val" / parts[2]
                if (legacy / "model_statedict").is_dir():
                    return legacy

        trained_root = resolve_vendor_models_path("pridict2", "trained_models")
        parts = name.split("/")
        if len(parts) == 3:
            resolved = trained_root / parts[0] / parts[1] / "train_val" / parts[2]
        else:
            resolved = trained_root / name
        if (resolved / "model_statedict").is_dir():
            return resolved
        raise ValueError(
            f"Unknown PRIDICT2 weights '{name}'. "
            f"Available: {self.list_available_weights()}"
        )

    def load_weights_by_name(self, name: str) -> None:
        """Load a named pre-trained PRIDICT2 weight set.

        Args:
            name: A weight set name from :meth:`list_available_weights`, or a
                path to a trained run directory.
        """
        self.load_model(str(self._resolve_weights_dir(name)))

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
        cell_types = kwargs.get('cell_types')
        if cell_types is None and self.is_trained and self.loaded_model_dir:
            cell_types = self._cell_types_from_loaded_config()
        model_name = self.model_name_str or kwargs.get('model_name', 'base_390k')

        # Prepare data using PRIDICT2's preprocessing pipeline
        dloader = self.prieml_model.prepare_data(
            df=df,
            model_name=model_name,
            cell_types=cell_types or [],
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

    def _build_datatensor(self, df: pd.DataFrame, y_ref: List[str]) -> tuple[Any, List[str]]:
        norm_cols, proc, init, n_init, mut, n_mut = self.prieml_model._process_df(df)
        dtensor = self.prieml_model._construct_datatensor(
            norm_cols, proc, init, n_init, mut, n_mut, y_ref=y_ref
        )
        return dtensor, list(norm_cols or [])

    def _run_train_val_once(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        y_ref: List[str],
        hyperparameters: Dict[str, Any],
        output_dir: str,
        run_suffix: str,
        trainer_backend: str = "legacy",
    ) -> Dict[str, Any]:
        from torch import nn
        from pridict2.pridict.pridictv2.run_workflow import build_config_map, train_val_run

        dtensor_train, norm_cols_tr = self._build_datatensor(train_df, y_ref)
        dtensor_val, _ = self._build_datatensor(val_df, y_ref)
        data_partitions = {0: {"train": dtensor_train, "validation": dtensor_val}}
        run_gpu_map = {0: int(hyperparameters.get("gpu_index", 0))}
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
            "seqlevel_featdim": int(hyperparameters.get("seqlevel_featdim", len(norm_cols_tr))),
            "num_outcomes": int(hyperparameters.get("num_outcomes", len(y_ref))),
        }
        config_map = build_config_map(
            trf_tup,
            experiment_options,
            loss_func=str(hyperparameters.get("loss_func", "KLDloss")),
        )
        run_output_dir = f"{output_dir}/{run_suffix}"
        train_val_run(
            data_partitions,
            config_map,
            run_output_dir,
            run_gpu_map,
            None,
            num_epochs,
            trainer_backend,
        )
        model_dir = f"{run_output_dir}/train_val/run_0"
        self.load_model(model_dir)
        prepared_val = self.prepare_data(val_df, y_ref=y_ref)
        pred_df = self._predict_from_loaded_or_current_model(dloader=prepared_val, y_ref=y_ref)
        fold_metrics: Dict[str, float] = {}
        for outcome in y_ref:
            true_col = f"true_{outcome}"
            pred_col = f"pred_{outcome}"
            if true_col not in pred_df.columns or pred_col not in pred_df.columns:
                continue
            y_true = pd.Series(pd.to_numeric(pred_df[true_col], errors="coerce"), index=pred_df.index)
            y_pred = pd.Series(pd.to_numeric(pred_df[pred_col], errors="coerce"), index=pred_df.index)
            mask = ~(y_true.isna().to_numpy() | y_pred.isna().to_numpy())
            y_true_clean = y_true.to_numpy(dtype=np.float64)[mask]
            y_pred_clean = y_pred.to_numpy(dtype=np.float64)[mask]
            if len(y_true_clean) == 0:
                fold_metrics[f"{outcome}_pearson"] = float("nan")
                fold_metrics[f"{outcome}_spearman"] = float("nan")
                continue
            corr = pearson_spearman(y_true_clean.tolist(), y_pred_clean.tolist())
            fold_metrics[f"{outcome}_pearson"] = float(corr["pearson"])
            fold_metrics[f"{outcome}_spearman"] = float(corr["spearman"])

        return {
            "output_dir": run_output_dir,
            "model_dir": model_dir,
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "metrics": fold_metrics,
        }
    
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
        hyperparameters = hyperparameters or {}
        trainer_backend = str(
            hyperparameters.get("trainer_backend", "pytorch-lightning")
        ).strip().lower()
        if trainer_backend in {"pytorch-lightning", "lightning", "pl"}:
            trainer_backend = "pytorch-lightning"
        else:
            trainer_backend = "legacy"
        y_ref = list(hyperparameters.get("y_ref", self._default_outcomes()) or self._default_outcomes())
        train_native, val_native = resolve_train_val_from_splits(train_data, val_data)
        train_df = self._to_pridict_dataframe(train_native)
        val_df = self._to_pridict_dataframe(val_native)

        output_dir = str(hyperparameters.get("output_dir", "artifacts/pridict2_train"))
        fold_reports: List[Dict[str, Any]] = []

        if val_data is None and has_assigned_cv_folds(train_data):
            for fold_idx, (fold_label, fold_train_native, fold_val_native) in enumerate(
                iter_assigned_cv_folds(train_data)
            ):
                fold_train_df = self._to_pridict_dataframe(fold_train_native)
                fold_val_df = self._to_pridict_dataframe(fold_val_native)
                report = self._run_train_val_once(
                    train_df=fold_train_df,
                    val_df=fold_val_df,
                    y_ref=y_ref,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    run_suffix=f"cv_{fold_label}",
                    trainer_backend=trainer_backend,
                )
                fold_reports.append({"fold": fold_idx, "fold_label": fold_label, **report})

        run_report = self._run_train_val_once(
            train_df=train_df,
            val_df=val_df,
            y_ref=y_ref,
            hyperparameters=hyperparameters,
            output_dir=output_dir,
            run_suffix="final",
            trainer_backend=trainer_backend,
        )
        result: Dict[str, Any] = {
            "status": "success",
            "output_dir": run_report["output_dir"],
            "model_dir": run_report["model_dir"],
            "num_train_rows": int(run_report["num_train_rows"]),
            "num_val_rows": int(run_report["num_val_rows"]),
            "outcomes": y_ref,
            "validation_metrics": run_report["metrics"],
        }
        if fold_reports:
            result["cross_validation"] = fold_reports
        return result
    
    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        """
        Evaluate PRIDICT2 model on all three outcomes using a registered weight set.

        Args:
            test_data: Test DataFrame with true labels
            weights: Registered weight set ID (see :meth:`list_available_weights`).

        Returns:
            Dictionary with evaluation metrics for each outcome
        """
        if not weights or not str(weights).strip():
            raise ValueError(
                "weights is required for evaluate(). "
                f"Available: {self.list_available_weights()}"
            )
        self.load_weights_by_name(weights)
        
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
                corr = pearson_spearman(y_true_clean.tolist(), y_pred_clean.tolist())
                pearson_corr = float(corr["pearson"])
                spearman_corr = float(corr["spearman"])
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
        """Copy the loaded PRIDICT2 run directory into ``model_path``."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        if not self.loaded_model_dir:
            raise ValueError("No loaded PRIDICT2 run directory to save.")

        import os
        import shutil

        src = Path(self.loaded_model_dir)
        os.makedirs(model_path, exist_ok=True)
        for name in ("model_statedict", "config"):
            src_sub = src / name
            if src_sub.is_dir():
                dest_sub = Path(model_path) / name
                if dest_sub.exists():
                    shutil.rmtree(dest_sub)
                shutil.copytree(src_sub, dest_sub)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name_str,
            'wsize': self.wsize,
            'available_weights': self.list_available_weights(),
            'outcomes': self._default_outcomes(),
            'supports_standardized_input': True,
            'description': 'PRIDICT2 model for predicting outcome distribution (edited/unedited/indel)'
        })
        if self.is_trained and self.loaded_model_dir:
            info['cell_types_from_config'] = self._cell_types_from_loaded_config()
        return info

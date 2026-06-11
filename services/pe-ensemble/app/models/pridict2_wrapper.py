import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path
from . import weights_registry
from pe_common.training import regression_metrics
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
        self.selected_cell_type: Optional[str] = None

    def _default_outcomes(self) -> List[str]:
        # Original PRIDICT1 base models were mostly used for intended edits.
        if self.model_name_str in {None, "base_90k", "base_390k"}:
            return ["averageedited"]
        return ["averageedited", "averageunedited", "averageindel"]

    @staticmethod
    def _normalize_cell_type(name: str) -> str:
        return "".join(str(name).split("_"))

    @staticmethod
    def _cell_types_from_run_dir(model_path: str) -> List[str]:
        """Return prediction heads available on disk for a run directory."""
        state_dict_dir = Path(model_path) / "model_statedict"
        return sorted(
            PRIDICT2ModelWrapper._normalize_cell_type(path.stem.replace("decoder_", ""))
            for path in state_dict_dir.glob("decoder_*.pkl")
        )

    def _cell_types_from_loaded_config(self) -> List[str]:
        """Return prediction-head cell types from the loaded weight run config."""
        if not self.loaded_model_dir:
            raise ValueError("PRIDICT2 weights are not loaded.")
        import os

        mconfig_dir = os.path.join(self.loaded_model_dir, "config")
        _, options = self.prieml_model._load_model_config(mconfig_dir)
        datasets = options.get("datasets_name") or []
        return [self._normalize_cell_type(name) for name in datasets]

    @staticmethod
    def _registered_base_weight_ids() -> List[str]:
        return weights_registry.list_weight_ids("pridict2")

    @staticmethod
    def _split_weight_name(name: str) -> tuple[str, Optional[str]]:
        """Split ``{base_run_id}`` or ``{base_run_id}__{cell_type}``."""
        candidate = name.strip()
        if not candidate:
            raise ValueError("PRIDICT2 weight name is empty.")

        base_ids = PRIDICT2ModelWrapper._registered_base_weight_ids()
        for base_id in sorted(base_ids, key=len, reverse=True):
            if candidate == base_id:
                return base_id, None
            prefix = f"{base_id}__"
            if candidate.startswith(prefix):
                cell_type = candidate[len(prefix):]
                if cell_type:
                    return base_id, cell_type
                break

        if Path(candidate).expanduser().is_dir():
            return candidate, None

        raise ValueError(
            f"Unknown PRIDICT2 weights '{name}'. "
            f"Available: {PRIDICT2ModelWrapper.list_available_weights()}"
        )

    @staticmethod
    def resolve_weight_selection(name: str) -> tuple[Path, Optional[str]]:
        """Resolve a weight selection to a run directory and optional cell-type head."""
        candidate = name.strip()
        cell_type: Optional[str] = None
        base_ref = candidate

        if "__" in candidate:
            maybe_base, maybe_cell = PRIDICT2ModelWrapper._split_weight_name(candidate)
            base_ref = maybe_base
            cell_type = maybe_cell

        base_path = Path(base_ref).expanduser()
        if base_path.is_dir() and (base_path / "model_statedict").is_dir():
            run_dir = base_path.resolve()
        else:
            registry_id = base_ref.replace("/", "__")
            try:
                run_dir = weights_registry.resolve_dir("pridict2", registry_id)
            except ValueError:
                trained_root = resolve_vendor_models_path("pridict2", "trained_models")
                parts = base_ref.replace("\\", "/").split("/")
                if len(parts) == 3:
                    legacy = trained_root / parts[0] / parts[1] / "train_val" / parts[2]
                elif "__" in registry_id:
                    legacy_parts = registry_id.split("__")
                    if len(legacy_parts) == 3:
                        legacy = (
                            trained_root
                            / legacy_parts[0]
                            / legacy_parts[1]
                            / "train_val"
                            / legacy_parts[2]
                        )
                    else:
                        legacy = trained_root / registry_id
                else:
                    legacy = trained_root / registry_id
                if (legacy / "model_statedict").is_dir():
                    run_dir = legacy.resolve()
                else:
                    raise ValueError(
                        f"Unknown PRIDICT2 weights '{name}'. "
                        f"Available: {PRIDICT2ModelWrapper.list_available_weights()}"
                    ) from None

        PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))
        available = PRIDICT2ModelWrapper._cell_types_from_run_dir(str(run_dir))
        if not available:
            raise ValueError(f"PRIDICT2 run has no decoder heads: {run_dir}")

        if cell_type is None:
            if len(available) == 1:
                return run_dir, available[0]
            raise ValueError(
                "PRIDICT2 weight selection must include a cell-type head suffix "
                f"for multi-head runs. Choose one of: "
                f"{[f'{base_ref}__{head}' for head in available]}"
            )

        normalized = PRIDICT2ModelWrapper._normalize_cell_type(cell_type)
        if normalized not in available:
            raise ValueError(
                f"PRIDICT2 head '{cell_type}' is not available for '{base_ref}'. "
                f"Available heads: {available}"
            )
        return run_dir, normalized

    def _to_pridict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` is already in PRIDICT's native schema.

        Standardized -> PRIDICT conversion is owned by the PE-DB service; fetch
        model-format data from ``GET /api/filter?...&format=pridict2`` rather than
        passing standardized rows here.
        """
        if self.PRIDICT_REQUIRED_COLUMNS.issubset(df.columns):
            out = df.copy()
            if "Correction_Length" in out.columns:
                out["Correction_Length"] = pd.to_numeric(
                    out["Correction_Length"], errors="raise"
                ).astype(int)
            return out
        missing = sorted(self.PRIDICT_REQUIRED_COLUMNS.difference(df.columns))
        raise ValueError(
            "PRIDICT2 expects native input columns; missing: "
            f"{missing}. Fetch model-format data from PE-DB "
            "(GET /api/filter?...&format=pridict2)."
        )

    def _predict_from_loaded_or_current_model(
        self, dloader: Any, y_ref: List[str]
    ) -> pd.DataFrame:
        if self.selected_cell_type and self.loaded_model_dir:
            return self.prieml_model.predict_from_dloader(
                dloader=dloader,
                model_dir=self.loaded_model_dir,
                y_ref=y_ref,
            )
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
    def _dataset_component_names(
        datasets_name: List[str],
        *,
        separate_attention_layers: bool,
        separate_seqlevel_embedder: bool,
    ) -> List[str]:
        """Return dataset-specific statedict filenames expected for a run config."""
        names: List[str] = []
        for raw_name in datasets_name:
            dname = "".join(str(raw_name).split("_"))
            names.append(f"decoder_{dname}.pkl")
            if separate_seqlevel_embedder:
                names.append(f"seqlevel_featembeder_{dname}.pkl")
            if separate_attention_layers:
                for seq_type in ("init", "mut"):
                    for attn_type in ("local", "global"):
                        names.append(f"{attn_type}_featemb_{seq_type}_attn_{dname}.pkl")
        return names

    @staticmethod
    def _validate_run_dir(model_path: str) -> None:
        """Ensure run config datasets match on-disk statedict component names."""
        import os
        import pickle

        config_dir = os.path.join(model_path, "config")
        state_dict_dir = os.path.join(model_path, "model_statedict")
        if not os.path.isdir(config_dir) or not os.path.isdir(state_dict_dir):
            raise ValueError(f"Invalid PRIDICT2 run directory: {model_path}")

        exp_options_path = os.path.join(config_dir, "exp_options.pkl")
        if not os.path.isfile(exp_options_path):
            raise ValueError(f"PRIDICT2 run config missing exp_options.pkl: {model_path}")

        with open(exp_options_path, "rb") as handle:
            options = pickle.load(handle)

        datasets = options.get("datasets_name") or []
        required = PRIDICT2ModelWrapper._dataset_component_names(
            datasets,
            separate_attention_layers=bool(options.get("separate_attention_layers")),
            separate_seqlevel_embedder=bool(options.get("separate_seqlevel_embedder")),
        )
        missing = [
            name
            for name in required
            if not os.path.isfile(os.path.join(state_dict_dir, name))
        ]
        if not missing:
            return

        on_disk = sorted(
            path.name
            for path in Path(state_dict_dir).glob("*.pkl")
            if path.name != "best_epoch.pkl"
            and any(
                token in path.name
                for token in ("decoder_", "seqlevel_featembeder_", "_featemb_")
            )
        )
        raise ValueError(
            "PRIDICT2 weight bundle is incomplete: "
            f"config expects datasets {datasets} but statedict is missing "
            f"{missing}. On-disk dataset-specific files: {on_disk}. "
            "This usually means the wrong model_statedict was packaged with the "
            "run config. Re-migrate from vendor or choose a compatible weight set."
        )

    @staticmethod
    def list_available_weights() -> List[str]:
        """List registered PRIDICT2 weight IDs, with cell-type head suffix when needed."""
        return [entry["id"] for entry in PRIDICT2ModelWrapper.list_available_weight_entries()]

    @staticmethod
    def list_available_weight_entries() -> List[Dict[str, Any]]:
        """List loadable PRIDICT2 weight entries for API/UI selection."""
        registry_by_id = {
            entry["id"]: entry for entry in weights_registry.list_entries("pridict2")
        }
        entries: List[Dict[str, Any]] = []
        for base_id in PRIDICT2ModelWrapper._registered_base_weight_ids():
            run_dir = weights_registry.resolve_dir("pridict2", base_id)
            try:
                PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))
            except ValueError:
                continue

            manifest = dict(registry_by_id.get(base_id, {}))
            base_label = manifest.get("label", base_id.replace("__", " / "))
            cell_types = PRIDICT2ModelWrapper._cell_types_from_run_dir(str(run_dir))
            if len(cell_types) == 1:
                entries.append(
                    {
                        **manifest,
                        "id": base_id,
                        "model": "pridict2",
                        "label": base_label,
                        "cell_type": cell_types[0],
                    }
                )
                continue

            for cell_type in cell_types:
                weight_id = f"{base_id}__{cell_type}"
                entries.append(
                    {
                        **manifest,
                        "id": weight_id,
                        "model": "pridict2",
                        "label": f"{base_label} / {cell_type}",
                        "cell_type": cell_type,
                    }
                )
        return entries

    def _resolve_weights_dir(self, name: str) -> Path:
        """Resolve a weight set ID (or directory path) to a run directory."""
        run_dir, cell_type = self.resolve_weight_selection(name)
        self.selected_cell_type = cell_type
        return run_dir

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
        self._validate_run_dir(model_path)
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
        if cell_types is None and self.selected_cell_type:
            cell_types = [self.selected_cell_type]
        elif cell_types is None and self.is_trained and self.loaded_model_dir:
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
            y_true = pd.to_numeric(pred_df[true_col], errors="coerce").to_numpy(dtype=np.float64)
            y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
            fold_metrics.update(regression_metrics(y_true, y_pred, prefix=outcome))

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
        from ..training.progress_log import take_job_training_callbacks

        hyperparameters, _progress_log, cancel_check = take_job_training_callbacks(hyperparameters)
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
                if cancel_check is not None:
                    cancel_check()
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

        if cancel_check is not None:
            cancel_check()
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
        if self.selected_cell_type and "dataset_name" in pred_df.columns:
            pred_df = pred_df[
                pred_df["dataset_name"] == self.selected_cell_type
            ].reset_index(drop=True)
        
        primary_outcome = "averageedited" if "averageedited" in outcomes else outcomes[0]
        results: Dict[str, float] = {}
        for outcome in outcomes:
            y_true = pred_df[f"true_{outcome}"].values
            y_pred = pred_df[f"pred_{outcome}"].values
            results.update(regression_metrics(y_true, y_pred, prefix=outcome))
            if outcome == primary_outcome:
                results.update(regression_metrics(y_true, y_pred))
        results["n_samples"] = int(len(pred_df))
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
        if self.selected_cell_type:
            info['selected_cell_type'] = self.selected_cell_type
        return info

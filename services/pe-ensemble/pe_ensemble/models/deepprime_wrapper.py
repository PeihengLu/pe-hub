# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
import sys
import json
import os
import math
from typing import List, Dict, Any, Optional, Tuple, cast
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

from pathlib import Path

from .vendor_path import resolve_vendor_models_path
from . import weights_registry

# Ensure vendor models are importable in local development
_vendor_root = resolve_vendor_models_path()
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))

from pe_common.model_interface import BasePEModel
from pe_common.training import (
    apply_fine_tune_freezing,
    build_lr_scheduler,
    dataloader_kwargs,
    fit_lightning_module,
    LightningTrainerConfig,
    regression_metrics,
    resolve_training_seed,
    seed_training_run,
)
from pe_common.splits import resolve_train_val_from_splits

from ..training.progress_log import log_training_best, make_epoch_logger, take_job_training_callbacks
from .hparams import (
    iter_cv_training_folds,
    require_evaluate_weights,
    resolve_pretrained_weight_id,
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
            max_epochs=int(self.hparams_map.get("epochs", 5)),
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class DeepPrimeModelWrapper(BasePEModel):
    """Wrapper for DeepPrime model"""

    DEEPPRIME_REQUIRED_COLUMNS = {
        'WT74_On', 'Edited74_On', 'PBSlen', 'RTlen', 'RT-PBSlen', 'Edit_pos',
        'Edit_len', 'RHA_len', 'type_sub', 'type_ins', 'type_del', 'Tm1', 'Tm2',
        'Tm2new', 'Tm3', 'Tm4', 'TmD', 'nGCcnt1', 'nGCcnt2', 'nGCcnt3',
        'fGCcont1', 'fGCcont2', 'fGCcont3', 'MFE3', 'MFE4', 'DeepSpCas9_score',
    }

    # Architecture of every DeepPrime vendor checkpoint.
    DEFAULT_ARCHITECTURE: Dict[str, int] = {"hidden_size": 128, "num_layers": 1}
    ARCHITECTURE_FILENAME = "architecture.json"
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        pe_system: Optional[str] = None,
        cell_type: Optional[str] = None,
    ):
        """
        Initialize DeepPrime model wrapper.

        Args:
            device: PyTorch device
            pe_system: Optional vendor PE system for legacy ``load_deepprime`` lookup
                when training with ``load_pretrained=True`` and no explicit weights.
            cell_type: Optional vendor cell type paired with ``pe_system`` for the
                same legacy lookup path.
        """
        super().__init__('DeepPrime', device)

        self.pe_system = pe_system
        self.cell_type = cell_type
        self.model_dir = None
        self.model_type = None
        self.models = []
        self.mean: Optional[pd.Series] = None
        self.std: Optional[pd.Series] = None
        # Constructor kwargs for GeneInteractionModel. Vendor checkpoints all use
        # the defaults below; from-scratch runs may override them, so they are
        # persisted with the checkpoint and read back on load.
        self.architecture: Dict[str, int] = dict(self.DEFAULT_ARCHITECTURE)
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
        """Read measured editing efficiency for a split, in its original units."""
        if "editing_efficiency" in source_df.columns:
            series = pd.Series(pd.to_numeric(source_df["editing_efficiency"], errors="coerce"), index=source_df.index)
        elif "Efficiency" in feature_df.columns:
            series = pd.Series(pd.to_numeric(feature_df["Efficiency"], errors="coerce"), index=feature_df.index)
        else:
            raise ValueError(
                f"{split_name} data must include 'editing_efficiency' or 'Efficiency' column."
            )
        n_missing = int(series.isna().sum())
        if n_missing:
            raise ValueError(
                f"{split_name} data has {n_missing} row(s) with a missing/non-numeric "
                "efficiency label. Filter these rows out upstream rather than training "
                "on them; imputing them as 0 would teach the model that unmeasured "
                "edits are inefficient."
            )
        return series.to_numpy(dtype=np.float32)

    @staticmethod
    def _to_model_space(efficiency: np.ndarray) -> np.ndarray:
        """Map measured efficiency into the space DeepPrime checkpoints regress on.

        Vendor inference ends with ``exp(pred) - 1`` (``vendor/models/deepprime``,
        ``src/dprime.py``), so the published checkpoints emit ``log1p(efficiency)``.
        Training must use the same target space, otherwise fine-tuning drags a
        pretrained head off its output scale and ``predict`` exponentiates a value
        that is already in efficiency units.
        """
        return np.log1p(np.clip(efficiency, 0.0, None)).astype(np.float32)

    def _init_trainable_models(self, hyperparameters: Dict[str, Any]) -> None:
        load_pretrained = bool(hyperparameters.get("load_pretrained", False))
        if load_pretrained:
            weight_id = resolve_pretrained_weight_id(hyperparameters)
            if weight_id:
                self.load_weights_by_name(weight_id)
            else:
                self.load_model()
            return
        from deepprime.src.dprime import GeneInteractionModel

        hidden_size = int(hyperparameters.get("hidden_size", 128))
        num_layers = int(hyperparameters.get("num_layers", 1))
        self.models = [
            GeneInteractionModel(hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        ]
        self.model = self.models
        self.architecture = {"hidden_size": hidden_size, "num_layers": num_layers}
        # Drop any stats carried over from a previous fit so each from-scratch
        # run refits normalization on its own train split (CV folds included).
        self.mean = None
        self.std = None
        self.is_trained = True

    def _fit_feature_normalization(self, train_feature_df: pd.DataFrame) -> None:
        """Fit tabular-feature z-scoring statistics on a training split.

        ``prepare_data`` z-scores the tabular features whenever ``mean``/``std``
        are set, and vendor checkpoints ship ``mean.csv``/``std.csv`` for exactly
        that reason. Training from scratch has no vendor stats file, so the
        statistics are derived here from the *training* split only (never val or
        test) and then persisted next to the checkpoint by ``save_model``.
        Without this, a from-scratch model trains on raw features but later
        reloads against DeepPrime's vendor stats, silently z-scoring inputs with
        statistics that were never applied during training.
        """
        from deepprime.src.utils import select_cols

        x_features = select_cols(train_feature_df).astype(float)
        self.mean = x_features.mean().astype(float)
        # Zero-variance columns would divide by zero; keep them as pass-through.
        self.std = x_features.std().replace(0.0, 1.0).fillna(1.0).astype(float)

    def _fit_models_on_split(
        self,
        *,
        train_source: pd.DataFrame,
        val_source: pd.DataFrame,
        hyperparameters: Dict[str, Any],
        progress_log: Any,
        cancel_check: Any,
        run_label: str,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, float]]:
        epochs = int(hyperparameters.get("epochs", 5))
        batch_size = int(hyperparameters.get("batch_size", 128))
        lr = float(hyperparameters.get("lr", 1e-4))
        weight_decay = float(hyperparameters.get("weight_decay", 0.0))
        freezing = bool(hyperparameters.get("freezing", False))
        grad_clip = float(hyperparameters.get("grad_clip", 1.0))
        scheduler_name = hyperparameters.get("scheduler", "none")
        scheduler_kwargs = hyperparameters.get("scheduler_kwargs", None)
        early_stopping_patience = int(hyperparameters.get("early_stopping_patience", 10))
        early_stopping_min_delta = float(hyperparameters.get("early_stopping_min_delta", 0.0))
        reshuffle_each_epoch = bool(hyperparameters.get("reshuffle_each_epoch", True))
        seed = resolve_training_seed(hyperparameters)

        train_feature_df = self._to_deepprime_feature_df(train_source)
        val_feature_df = self._to_deepprime_feature_df(val_source)
        y_train = self._extract_targets(train_source, train_feature_df, "Training")
        y_val = self._extract_targets(val_source, val_feature_df, "Validation")
        if self.mean is None or self.std is None:
            self._fit_feature_normalization(train_feature_df)
        train_inputs = self.prepare_data(train_feature_df)
        val_inputs = self.prepare_data(val_feature_df)
        loader_kwargs = dataloader_kwargs(
            num_workers=hyperparameters.get("num_workers"),
            pin_memory=self.device.type == "cuda",
        )
        train_loader = DataLoader(
            _DeepPrimeTensorDataset(
                train_inputs["g"], train_inputs["x"], self._to_model_space(y_train)
            ),
            batch_size=batch_size,
            shuffle=reshuffle_each_epoch,
            drop_last=False,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            _DeepPrimeTensorDataset(
                val_inputs["g"], val_inputs["x"], self._to_model_space(y_val)
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        history: List[Dict[str, float]] = []
        model_summaries: List[Dict[str, float]] = []
        for model_idx, model in enumerate(self.models):
            if freezing:
                apply_fine_tune_freezing(model, model.head)
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
                    # Offset per ensemble member so a from-scratch ensemble still
                    # gets diverse initializations under a fixed seed.
                    seed=seed + model_idx if seed is not None else None,
                ),
                on_epoch_end=make_epoch_logger(
                    progress_log,
                    prefix=f"{run_label}model {model_idx} |",
                    cancel_check=cancel_check,
                ),
            )
            if int(metrics["best_epoch"]) < 0 or not np.isfinite(float(metrics["best_val_loss"])):
                raise ValueError(
                    "DeepPrime training finished without a finite validation loss "
                    f"for model {model_idx} "
                    f"(history_epochs={len(metrics.get('history') or [])}). "
                    "Check for non-finite features or labels in the training split."
                )
            log_training_best(
                progress_log,
                best_epoch=int(metrics["best_epoch"]),
                best_val_loss=float(metrics["best_val_loss"]),
                prefix=f"{run_label}model {model_idx} |",
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

        # Score the ensemble in efficiency units so the numbers are comparable
        # with OPED/PRIDICT2 and with `evaluate()` on the registered weights.
        val_metrics = regression_metrics(y_val, np.asarray(self.predict(val_inputs), dtype=float))
        return history, model_summaries, val_metrics

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load pre-trained DeepPrime model

        Args:
            model_path: Optional registry entry directory or legacy vendor path.
                If None, loads the default variant for ``pe_system``/``cell_type``.
        """
        from glob import glob
        from deepprime.models.load_model import load_deepprime
        from deepprime.src.dprime import GeneInteractionModel

        if model_path is None:
            if not self.pe_system or not self.cell_type:
                raise ValueError(
                    "DeepPrime.load_model() requires a weight path/ID, or both "
                    "pe_system and cell_type for vendor default lookup."
                )
            _, model_type = load_deepprime(
                self.pe_system,
                self.cell_type,
                silent=True,
            )
            self.load_weights_by_name(model_type)
            return

        entry = Path(str(model_path)).expanduser()
        if entry.is_dir() and (entry / "mean.csv").is_file() and list(entry.glob("*.pt")):
            self.model_dir = str(entry)
            self.model_type = entry.name
            mean_path = entry / "mean.csv"
            std_path = entry / "std.csv"
            model_files = sorted(entry.glob("*.pt"))
        elif model_path.endswith(".pt"):
            self.model_dir = os.path.dirname(os.path.dirname(model_path))
            self.model_type = os.path.basename(os.path.dirname(model_path))
            mean_path = Path(f"{self.model_dir}/DeepPrime_base/mean.csv")
            std_path = Path(f"{self.model_dir}/DeepPrime_base/std.csv")
            model_files = glob(f"{self.model_dir}/{self.model_type}/*.pt")
        else:
            self.model_dir = os.path.dirname(str(model_path))
            self.model_type = os.path.basename(str(model_path).rstrip("/"))
            mean_path = Path(f"{self.model_dir}/DeepPrime_base/mean.csv")
            std_path = Path(f"{self.model_dir}/DeepPrime_base/std.csv")
            model_files = glob(f"{self.model_dir}/{self.model_type}/*.pt")

        mean_obj = pd.read_csv(mean_path, header=None, index_col=0).squeeze()
        std_obj = pd.read_csv(std_path, header=None, index_col=0).squeeze()
        self.mean = cast(pd.Series, mean_obj if isinstance(mean_obj, pd.Series) else pd.Series(dtype=float))
        self.std = cast(pd.Series, std_obj if isinstance(std_obj, pd.Series) else pd.Series(dtype=float))

        if not model_files:
            raise FileNotFoundError(
                f"No model files found for DeepPrime weights at {model_path}"
            )

        self.architecture = self._read_architecture(Path(str(model_files[0])).parent)
        self.models = []
        for m_path in model_files:
            model = GeneInteractionModel(**self.architecture).to(self.device)
            model.load_state_dict(
                torch.load(m_path, map_location=torch.device(self.device))
            )
            model.eval()
            self.models.append(model)

        self.model = self.models
        self.is_trained = True
    
    @classmethod
    def _read_architecture(cls, entry_dir: Path) -> Dict[str, int]:
        """Read persisted constructor kwargs, falling back to the vendor defaults."""
        arch_path = entry_dir / cls.ARCHITECTURE_FILENAME
        if not arch_path.is_file():
            return dict(cls.DEFAULT_ARCHITECTURE)
        try:
            with open(arch_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable {arch_path}: {exc}") from exc
        return {
            key: int(payload.get(key, default))
            for key, default in cls.DEFAULT_ARCHITECTURE.items()
        }

    @staticmethod
    def list_available_weights() -> List[str]:
        """List registered DeepPrime weight set IDs."""
        return weights_registry.list_weight_ids("deepprime")

    def load_weights_by_name(self, name: str) -> None:
        """Load a registered DeepPrime weight set by ID or directory path."""
        candidate = Path(name).expanduser()
        if candidate.is_dir():
            self.load_model(str(candidate))
            return

        try:
            entry_dir = weights_registry.resolve_dir("deepprime", name)
            self.load_model(str(entry_dir))
            return
        except ValueError:
            pass

        models_root = resolve_vendor_models_path("deepprime", "models", "DeepPrime")
        legacy = models_root / name
        if legacy.is_dir():
            self.load_model(str(legacy))
            return
        raise ValueError(
            f"Unknown DeepPrime weights '{name}'. "
            f"Available: {self.list_available_weights()}"
        )

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
        
        # Collect predictions from ensemble. eval() matters when predicting right
        # after a fit, since Lightning leaves the module in training mode and
        # DeepPrime's head contains dropout.
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(g_tensor, x_tensor).detach().cpu().numpy()
            preds.append(pred)
        
        # Average over the ensemble axis only. Reshaping to (n_models, n_rows)
        # first is deliberate: np.squeeze() would drop the model axis for a
        # single-model ensemble (the from-scratch training case) and collapse the
        # per-row means into one scalar.
        stacked = np.asarray(preds, dtype=float).reshape(len(self.models), -1)
        mean_preds = stacked.mean(axis=0)

        # Invert the log1p target transform the checkpoints were trained on.
        return (np.exp(mean_preds) - 1).tolist()
    
    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fine-tune loaded DeepPrime model(s) on prepared data.

        This is a lightweight fine-tuning path (not the original DeepPrime
        training pipeline). Inputs must already be in DeepPrime's native feature
        schema (fetch model-format data from PE-DB: /api/filter?...&format=deepprime).
        """
        hyperparameters, progress_log, cancel_check = take_job_training_callbacks(hyperparameters)
        seed_training_run(hyperparameters)
        epochs = int(hyperparameters.get("epochs", 5))
        batch_size = int(hyperparameters.get("batch_size", 128))
        lr = float(hyperparameters.get("lr", 1e-4))
        load_pretrained = bool(hyperparameters.get("load_pretrained", False))
        freezing = bool(hyperparameters.get("freezing", False))

        source_df = train_data.reset_index(drop=True)
        cv_reports: List[Dict[str, Any]] = []
        for fold_idx, fold_label, fold_train, fold_val in iter_cv_training_folds(
            source_df, val_data, cancel_check=cancel_check
        ):
            self._init_trainable_models(hyperparameters)
            _, fold_summaries, fold_val_metrics = self._fit_models_on_split(
                train_source=fold_train,
                val_source=fold_val,
                hyperparameters=hyperparameters,
                progress_log=progress_log,
                cancel_check=cancel_check,
                run_label=f"{fold_label} |",
            )
            fold_losses = [float(row["best_val_loss"]) for row in fold_summaries]
            cv_reports.append(
                {
                    "fold": fold_idx,
                    "fold_label": fold_label,
                    "n_train": int(len(fold_train)),
                    "n_val": int(len(fold_val)),
                    "best_val_loss": float(sum(fold_losses) / len(fold_losses))
                    if fold_losses
                    else float("nan"),
                    "val_pearson": float(fold_val_metrics["pearson"]),
                    "val_spearman": float(fold_val_metrics["spearman"]),
                    "model_summaries": fold_summaries,
                }
            )

        train_source, val_source = resolve_train_val_from_splits(source_df, val_data)
        self._init_trainable_models(hyperparameters)
        history, model_summaries, val_metrics = self._fit_models_on_split(
            train_source=train_source,
            val_source=val_source,
            hyperparameters=hyperparameters,
            progress_log=progress_log,
            cancel_check=cancel_check,
            run_label="final |",
        )

        self._last_training_history = history
        self.model = self.models
        self.is_trained = True
        final = history[-1] if history else {}
        result: Dict[str, Any] = {
            "status": "success",
            "n_models_trained": len(self.models),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "load_pretrained": load_pretrained,
            "freezing": freezing,
            "hidden_size": int(hyperparameters.get("hidden_size", 128)),
            "num_layers": int(hyperparameters.get("num_layers", 1)),
            "final_train_loss": final.get("train_loss"),
            "final_val_loss": final.get("val_loss"),
            "model_summaries": model_summaries,
            "val_pearson": float(val_metrics["pearson"]),
            "val_spearman": float(val_metrics["spearman"]),
            "validation_metrics": val_metrics,
        }
        if cv_reports:
            losses = [
                float(row["best_val_loss"])
                for row in cv_reports
                if row.get("best_val_loss") is not None
                and not math.isnan(float(row["best_val_loss"]))
            ]
            spearmans = [
                float(row["val_spearman"])
                for row in cv_reports
                if not math.isnan(float(row.get("val_spearman", float("nan"))))
            ]
            pearsons = [
                float(row["val_pearson"])
                for row in cv_reports
                if not math.isnan(float(row.get("val_pearson", float("nan"))))
            ]
            result["cross_validation"] = {
                "folds": cv_reports,
                "mean_best_val_loss": float(sum(losses) / len(losses)) if losses else float("nan"),
                "mean_val_pearson": float(sum(pearsons) / len(pearsons)) if pearsons else float("nan"),
                "mean_val_spearman": float(sum(spearmans) / len(spearmans)) if spearmans else float("nan"),
            }
        return result
    
    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        """
        Evaluate DeepPrime model using a registered weight set.

        Args:
            test_data: DataFrame with input features and 'Efficiency' column
            weights: Registered weight set ID (see :meth:`list_available_weights`).

        Returns:
            Dictionary with evaluation metrics (Pearson, Spearman)
        """
        weights = require_evaluate_weights(self, weights)
        self.load_weights_by_name(weights)

        if 'Efficiency' in test_data.columns:
            y_true = test_data['Efficiency'].values
            X_test = test_data.drop('Efficiency', axis=1)
        elif 'PE_efficiency' in test_data.columns:
            y_true = test_data['PE_efficiency'].values
            X_test = test_data.drop('PE_efficiency', axis=1)
        else:
            raise ValueError("Test data must contain 'Efficiency' or 'PE_efficiency' column")

        prepared_data = self.prepare_data(X_test)
        y_pred = np.array(self.predict(prepared_data))
        return regression_metrics(y_true, y_pred)
    
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

        # ``load_model`` needs both files to rebuild the exact feature scaling and
        # layer shapes this checkpoint was trained with.
        if self.mean is None or self.std is None:
            raise ValueError(
                "Refusing to save DeepPrime weights without feature normalization "
                "statistics: prepare_data() would z-score inference inputs with "
                "statistics that were never applied during training."
            )
        self.mean.to_csv(os.path.join(model_path, 'mean.csv'), header=False)
        self.std.to_csv(os.path.join(model_path, 'std.csv'), header=False)
        with open(
            os.path.join(model_path, self.ARCHITECTURE_FILENAME), "w", encoding="utf-8"
        ) as handle:
            json.dump(self.architecture, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def save_to_registry(self, dest_dir) -> str:
        self.save_model(str(dest_dir))
        return "deepprime_ensemble"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'pe_system': self.pe_system,
            'cell_type': self.cell_type,
            'n_models': len(self.models),
            'model_type': self.model_type,
            'supports_standardized_input': True,
            'available_weights': self.list_available_weights(),
        })
        return info

"""Shared training utilities for PE model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl  # type: ignore[reportMissingImports]
from lightning.pytorch.callbacks import Callback, EarlyStopping as LightningEarlyStopping  # type: ignore[reportMissingImports]


@dataclass
class EarlyStopping:
    """Simple early stopping helper for minimizing a scalar metric."""

    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_score: float = float("inf")
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Update state and return True when current score is best."""
        if score < (self.best_score - self.min_delta):
            self.best_score = score
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False


def ensure_lightning_available() -> None:
    """PyTorch Lightning is assumed to be installed."""
    return None


def lightning_accelerator_from_device(device: torch.device) -> str:
    """Map torch.device to lightning Trainer accelerator."""
    from .devices import device_to_lightning_accelerator

    return device_to_lightning_accelerator(device)


@dataclass
class LightningTrainerConfig:
    """Trainer-level config for shared Lightning fits."""

    max_epochs: int
    min_epochs: int = 1
    grad_clip: Optional[float] = None
    patience: Optional[int] = None
    min_delta: float = 0.0
    log_every_n_steps: int = 25
    enable_progress_bar: bool = False
    deterministic: bool = True


def _normalize_log_prefix(prefix: str) -> str:
    cleaned = prefix.strip()
    while cleaned.endswith("|"):
        cleaned = cleaned[:-1].strip()
    return cleaned


def format_epoch_metrics_row(row: Dict[str, float], *, prefix: str = "") -> str:
    """Format one epoch of training metrics for job logs."""
    epoch = int(row.get("epoch", -1))
    parts: List[str] = []
    if prefix:
        parts.append(_normalize_log_prefix(prefix))
    if "model_index" in row and np.isfinite(row["model_index"]):
        parts.append(f"model {int(row['model_index'])}")
    parts.append(f"epoch {epoch}")
    for key in ("train_loss", "val_loss", "val_pearson", "val_spearman"):
        if key not in row:
            continue
        value = float(row[key])
        if np.isfinite(value):
            parts.append(f"{key}={value:.6g}")
    return " | ".join(parts)


def fit_lightning_module(
    module: Any,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: LightningTrainerConfig,
    on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """
    Fit a LightningModule and return shared training diagnostics.

    Expects the module to expose the wrapped torch model at `module.model`.
    """
    ensure_lightning_available()

    class _LightningHistoryCallback(Callback):
        def __init__(self, epoch_callback: Optional[Callable[[Dict[str, float]], None]]) -> None:
            super().__init__()
            self._epoch_callback = epoch_callback
            self.history: List[Dict[str, float]] = []
            self.best_val_loss: float = float("inf")
            self.best_epoch: int = -1
            self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        @staticmethod
        def _as_float(metrics: Dict[str, Any], key: str) -> float:
            value = metrics.get(key)
            if value is None:
                return float("nan")
            if torch.is_tensor(value):
                return float(value.detach().cpu().item())
            return float(value)

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            metrics = dict(trainer.callback_metrics)
            epoch = int(trainer.current_epoch)
            train_loss = self._as_float(metrics, "train_loss")
            val_loss = self._as_float(metrics, "val_loss")
            row: Dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            for key in ("val_pearson", "val_spearman"):
                if key in metrics:
                    row[key] = self._as_float(metrics, key)
            self.history.append(row)
            if self._epoch_callback is not None:
                self._epoch_callback(dict(row))

            if np.isfinite(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in pl_module.model.state_dict().items()
                }

    history_callback = _LightningHistoryCallback(on_epoch_end)
    callbacks: List[Any] = [history_callback]
    if config.patience is not None and config.patience > 0:
        callbacks.append(
            LightningEarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(config.patience),
                min_delta=float(config.min_delta),
            )
        )

    trainer = pl.Trainer(
        accelerator=lightning_accelerator_from_device(device),
        devices=1,
        max_epochs=int(config.max_epochs),
        min_epochs=int(config.min_epochs),
        callbacks=callbacks,
        deterministic=bool(config.deterministic),
        gradient_clip_val=float(config.grad_clip) if config.grad_clip is not None else 0.0,
        log_every_n_steps=max(1, int(config.log_every_n_steps)),
        enable_progress_bar=bool(config.enable_progress_bar),
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if history_callback.best_state_dict is not None:
        module.model.load_state_dict(history_callback.best_state_dict)

    # Lightning moves the module to CPU on fit end. Restore the caller's device so
    # post-fit predict/eval (which often places inputs on ``device``) does not hit
    # embedding device mismatches.
    module.to(device)

    return {
        "history": history_callback.history,
        "best_epoch": int(history_callback.best_epoch),
        "best_val_loss": float(history_callback.best_val_loss),
        "n_epochs_ran": len(history_callback.history),
    }


def clip_gradients(model: torch.nn.Module, max_norm: Optional[float]) -> None:
    """Clip gradients in-place when max_norm is set."""
    if max_norm is None:
        return
    if float(max_norm) <= 0:
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))


def apply_fine_tune_freezing(
    model: torch.nn.Module,
    trainable_module: torch.nn.Module,
) -> None:
    """Freeze all model parameters except those in ``trainable_module``."""
    for param in model.parameters():
        param.requires_grad = False
    for param in trainable_module.parameters():
        param.requires_grad = True


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: Optional[str] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Construct a torch LR scheduler by name."""
    if not scheduler_name:
        return None
    scheduler_kwargs = scheduler_kwargs or {}
    name = str(scheduler_name).lower()
    if name == "step":
        step_size = int(scheduler_kwargs.get("step_size", 10))
        gamma = float(scheduler_kwargs.get("gamma", 0.95))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        t_max = int(scheduler_kwargs.get("t_max", 10))
        eta_min = float(scheduler_kwargs.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "exponential":
        gamma = float(scheduler_kwargs.get("gamma", 0.98))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")


def build_group_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    group_col: str = "group_id",
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build deterministic group-aware folds.

    If `group_col` is missing or all-null, falls back to row-based folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    n_rows = len(df)
    if n_rows == 0:
        return []

    if group_col in df.columns:
        groups = pd.Series(pd.to_numeric(df[group_col], errors="coerce"), index=df.index)
        if bool(groups.notna().any()):
            unique_groups = groups.dropna().astype(int).unique().tolist()
        else:
            unique_groups = []
        if len(unique_groups) >= n_splits:
            rng = np.random.default_rng(random_state)
            rng.shuffle(unique_groups)
            group_buckets: List[List[int]] = [[] for _ in range(n_splits)]
            for i, group_id in enumerate(unique_groups):
                group_buckets[i % n_splits].append(int(group_id))

            all_indices = np.arange(n_rows)
            folds: List[Tuple[np.ndarray, np.ndarray]] = []
            for bucket in group_buckets:
                val_mask = np.isin(groups.to_numpy(dtype=float), np.asarray(bucket, dtype=float))
                val_idx = all_indices[val_mask]
                train_idx = all_indices[~val_mask]
                if len(train_idx) > 0 and len(val_idx) > 0:
                    folds.append((train_idx, val_idx))
            if folds:
                return folds

    # Fallback row-based folds.
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    split_indices = np.array_split(indices, n_splits)
    folds = []
    for val_idx in split_indices:
        if len(val_idx) == 0:
            continue
        train_idx = np.array([i for i in indices if i not in set(val_idx.tolist())], dtype=int)
        if len(train_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def pearson_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute Pearson and Spearman correlations with pandas."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pearson = float(df.corr(method="pearson").iloc[0, 1]) if len(df) > 1 else float("nan")
    spearman = float(df.corr(method="spearman").iloc[0, 1]) if len(df) > 1 else float("nan")
    return {"pearson": pearson, "spearman": spearman}


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Standard regression metrics after dropping NaN pairs."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same length")
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_clean = y_true_arr[mask]
    y_pred_clean = y_pred_arr[mask]
    n = int(y_true_clean.size)
    nan = float("nan")
    if n < 2:
        base = {"pearson": nan, "spearman": nan, "mse": nan, "mae": nan, "n_samples": n}
    else:
        corr = pearson_spearman(y_true_clean.tolist(), y_pred_clean.tolist())
        diff = y_true_clean - y_pred_clean
        base = {
            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "mse": float(np.mean(diff**2)),
            "mae": float(np.mean(np.abs(diff))),
            "n_samples": n,
        }
    if not prefix:
        return base
    return {f"{prefix}_{key}": value for key, value in base.items() if key != "n_samples"}


def run_supervised_training_loop(
    model: torch.nn.Module,
    *,
    num_epochs: int,
    train_epoch_fn: Callable[[int], float],
    validate_epoch_fn: Callable[[int], Dict[str, float]],
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """
    Run a model training loop with reusable control flow.

    Expects `validate_epoch_fn` to return at least {"val_loss": float}.
    """
    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(int(num_epochs)):
        train_loss = float(train_epoch_fn(epoch))
        val_info = validate_epoch_fn(epoch)
        val_loss = float(val_info.get("val_loss", float("inf")))
        row: Dict[str, float] = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        for key, value in val_info.items():
            row[key] = float(value)
        history.append(row)
        if on_epoch_end is not None:
            on_epoch_end(dict(row))

        is_best = val_loss < best_val_loss
        if early_stopping is not None:
            is_best = early_stopping.step(val_loss)
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()
        if early_stopping is not None and early_stopping.should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "n_epochs_ran": len(history),
    }

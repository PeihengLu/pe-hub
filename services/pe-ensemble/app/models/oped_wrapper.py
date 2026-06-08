import sys
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

from .vendor_path import resolve_vendor_models_path
from . import weights_registry
from pe_common.model_interface import BasePEModel
from pe_common.constants import DEVICE, DATA_ROOT
from pe_common.sequence_utils import (
    reverse_complement,
    sanitize_dna_sequence,
)
from pe_common.training import (
    build_lr_scheduler,
    fit_lightning_module,
    LightningTrainerConfig,
    pearson_spearman,
)
from pe_common.splits import (
    has_assigned_cv_folds,
    iter_assigned_cv_folds,
    resolve_train_val_from_splits,
)

# Ensure that the OPED model code directory is in sys.path
_vendor_root = resolve_vendor_models_path()
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))


class _OPEDEncodedDataset(Dataset):
    def __init__(self, encoded_df: pd.DataFrame) -> None:
        feature_columns = [
            "Target",
            "PBS",
            "RT",
            "Target_o2",
            "PBS_o2",
            "RT_o2",
            "Target_o3",
            "PBS_o3",
            "RT_o3",
        ]
        self.features: List[np.ndarray] = [
            np.asarray(list(encoded_df[col]), dtype=np.int64) for col in feature_columns
        ]
        self.targets = encoded_df["Efficiency"].astype(float).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        features = tuple(
            torch.tensor(feature[idx], dtype=torch.long) for feature in self.features
        )
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return features, target


class _OPEDLightningRegressor(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.hparams_map = dict(hparams)
        self.criterion = torch.nn.MSELoss()

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        pred, _ = self.model(inputs)
        return pred.squeeze(-1)

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], torch.Tensor], _batch_idx: int) -> torch.Tensor:
        inputs, target = batch
        pred = self.forward(inputs)
        loss = self.criterion(pred, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        inputs, target = batch
        pred = self.forward(inputs)
        loss = self.criterion(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.hparams_map.get("lr", 3e-4)),
            weight_decay=float(self.hparams_map.get("weight_decay", 0.0)),
        )
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_name=self.hparams_map.get("scheduler", "step"),
            scheduler_kwargs=self.hparams_map.get("scheduler_kwargs", {"step_size": 10, "gamma": 0.95}),
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

class OPEDModelWrapper(BasePEModel):
    """Wrapper for OPED (Order-based Prediction of Editing outcomes and Deletion) model"""

    # OPED ships its trained weights inside the vendor submodule. We load the
    # state_dict variant (decoupled from the PyTorch version), never the legacy
    # full-pickle artifacts which embed the original module paths.
    DEFAULT_WEIGHTS_RELPATH = (
        "oped",
        "pegRNA_PredictingCodes",
        "Model_Trained",
        "pegRNA_Model_Merged_saved.order3_decoder_weights.pt",
    )
    # Per-order k-mer vocab sizes (order1/2/3). The trained embeddings have
    # ntokens+1 rows (the +1 is the padding index), i.e. [5, 17, 65].
    MODEL_NTOKENS = [4, 16, 64]

    def __init__(self, device: torch.device = torch.device(DEVICE)):
        super().__init__('OPED', device)
        self.model_dir = None

    DEFAULT_WEIGHT_ID = "pegRNA_Model_Merged_saved.order3_decoder_weights"

    @staticmethod
    def list_available_weights() -> List[str]:
        """List registered OPED weight set IDs."""
        return weights_registry.list_weight_ids("oped")

    def load_weights_by_name(self, name: str) -> None:
        """Load a registered OPED weight set by ID, file path, or entry directory."""
        candidate = Path(name).expanduser()
        if candidate.is_file():
            self.load_model(str(candidate))
            return
        if candidate.is_dir():
            weights_file = candidate / "weights.pt"
            if weights_file.is_file():
                self.load_model(str(weights_file))
                return

        try:
            entry_dir = weights_registry.resolve_dir("oped", name)
            self.load_model(str(entry_dir / "weights.pt"))
            return
        except ValueError:
            pass

        model_root = resolve_vendor_models_path(
            "oped", "pegRNA_PredictingCodes", "Model_Trained"
        )
        target = model_root / (name if name.endswith(".pt") else f"{name}.pt")
        if target.is_file():
            self.load_model(str(target))
            return
        raise ValueError(
            f"Unknown OPED weights '{name}'. Available: {self.list_available_weights()}"
        )

    def _resolve_weights_path(self, model_path: Optional[str]) -> str:
        """Resolve a usable OPED state_dict path.

        Accepts an explicit file, a directory (searched for the weights file),
        or None (uses the bundled default inside vendor/models).
        """
        if model_path is None:
            try:
                entry_dir = weights_registry.resolve_dir("oped", self.DEFAULT_WEIGHT_ID)
                return str(entry_dir / "weights.pt")
            except ValueError:
                return str(resolve_vendor_models_path(*self.DEFAULT_WEIGHTS_RELPATH))

        candidate = Path(model_path).expanduser()
        if candidate.is_dir():
            # Prefer the canonical state_dict file name if present, else the
            # first *_weights.pt in the directory.
            preferred = candidate / self.DEFAULT_WEIGHTS_RELPATH[-1]
            if preferred.is_file():
                return str(preferred)
            weight_files = sorted(candidate.glob("*_weights.pt"))
            if weight_files:
                return str(weight_files[0])
            raise FileNotFoundError(
                f"No OPED state_dict (*_weights.pt) found in directory {candidate}."
            )
        return str(candidate)

    def _load_state_dict(self, weights_path: str) -> Dict[str, torch.Tensor]:
        """Load an OPED state_dict, rejecting legacy full-pickle artifacts."""
        try:
            obj = torch.load(weights_path, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
            raise ValueError(
                f"Failed to load OPED weights as a state_dict from '{weights_path}'. "
                "OPED's legacy full-pickle files (e.g. '*.order3_decoder.pt' and "
                "'*_torch2.pt') are not compatible across PyTorch versions and must "
                "not be loaded directly. Convert them once with "
                "`python -m app.models.convert_oped_weights <pickle> <out_weights.pt>` "
                "and load the resulting state_dict instead."
            ) from exc

        if not isinstance(obj, dict) or not all(torch.is_tensor(v) for v in obj.values()):
            raise ValueError(
                f"File '{weights_path}' is not an OPED state_dict. Expected a dict of "
                "tensors; got a full pickled model. Convert it with "
                "`python -m app.models.convert_oped_weights` first."
            )
        return obj

    @staticmethod
    def _build_kmer_vocab(k: int) -> Dict[str, int]:
        if k <= 0:
            raise ValueError("k must be positive")

        vocab: Dict[str, int] = {}
        kmers = [""]
        alphabet = "ACGT"
        for _ in range(k):
            kmers = [prefix + base for prefix in kmers for base in alphabet]
        for idx, kmer in enumerate(kmers, start=1):
            vocab[kmer] = idx
        return vocab

    @staticmethod
    def _encode_kmers_with_padding(
        sequence: str, k: int, vocab: Dict[str, int], padded_len: int
    ) -> List[int]:
        padded_len = max(0, int(padded_len))
        sequence = sanitize_dna_sequence(sequence)
        if len(sequence) < k:
            return [0] * padded_len

        encoded = [vocab[sequence[i : i + k]] for i in range(0, len(sequence) - k + 1)]
        if len(encoded) < padded_len:
            encoded.extend([0] * (padded_len - len(encoded)))
        return encoded

    @staticmethod
    def _to_oped_numeric_df(df_oped: pd.DataFrame) -> pd.DataFrame:
        """Convert OPED sequences to encoded numeric format.

        Args:
            df_oped (pd.DataFrame): OPED sequences data

        Returns:
            pd.DataFrame: OPED sequences data in encoded numeric format
        """
        if df_oped.empty:
            empty_columns = pd.Index(
                [
                    "Target",
                    "PBS",
                    "RT",
                    "Target_o2",
                    "PBS_o2",
                    "RT_o2",
                    "Target_o3",
                    "PBS_o3",
                    "RT_o3",
                ]
            )
            return pd.DataFrame(
                columns=empty_columns
            )

        char2id = OPEDModelWrapper._build_kmer_vocab(1)
        char2id_o2 = OPEDModelWrapper._build_kmer_vocab(2)
        char2id_o3 = OPEDModelWrapper._build_kmer_vocab(3)

        target_series = df_oped["Target(47bp)"].astype(str).map(sanitize_dna_sequence)
        pbs_series = df_oped["PBS"].astype(str).map(sanitize_dna_sequence)
        rt_series = df_oped["RT"].astype(str).map(sanitize_dna_sequence)

        target_lengths = target_series.map(len).to_numpy(dtype=int)
        pbs_lengths = pbs_series.map(len).to_numpy(dtype=int)
        rt_lengths = rt_series.map(len).to_numpy(dtype=int)

        max_target_len = int(max(int(np.max(target_lengths)), 47))
        max_pbs_len = int(max(int(np.max(pbs_lengths)), 1))
        max_rt_len = int(max(int(np.max(rt_lengths)), 1))

        records = []
        for target, pbs, rt in zip(target_series, pbs_series, rt_series):
            pbs_rc = sanitize_dna_sequence(reverse_complement(pbs))
            rt_rc = sanitize_dna_sequence(reverse_complement(rt))
            target = sanitize_dna_sequence(target)

            record = {
                "Target": OPEDModelWrapper._encode_kmers_with_padding(
                    target, 1, char2id, max_target_len
                ),
                "PBS": OPEDModelWrapper._encode_kmers_with_padding(
                    pbs_rc, 1, char2id, max_pbs_len
                ),
                "RT": OPEDModelWrapper._encode_kmers_with_padding(
                    rt_rc, 1, char2id, max_rt_len
                ),
                "Target_o2": OPEDModelWrapper._encode_kmers_with_padding(
                    target, 2, char2id_o2, max_target_len - 1
                ),
                "PBS_o2": OPEDModelWrapper._encode_kmers_with_padding(
                    pbs_rc, 2, char2id_o2, max_pbs_len - 1
                ),
                "RT_o2": OPEDModelWrapper._encode_kmers_with_padding(
                    rt_rc, 2, char2id_o2, max_rt_len - 1
                ),
                "Target_o3": OPEDModelWrapper._encode_kmers_with_padding(
                    target, 3, char2id_o3, max_target_len - 2
                ),
                "PBS_o3": OPEDModelWrapper._encode_kmers_with_padding(
                    pbs_rc, 3, char2id_o3, max_pbs_len - 2
                ),
                "RT_o3": OPEDModelWrapper._encode_kmers_with_padding(
                    rt_rc, 3, char2id_o3, max_rt_len - 2
                ),
            }
            records.append(record)

        return pd.DataFrame(records)

    @staticmethod
    def _build_cache_path(
        df: pd.DataFrame,
        target_len: int,
    ) -> Path:
        cache_dir = Path(DATA_ROOT) / "formatted" / "oped"
        cache_dir.mkdir(parents=True, exist_ok=True)

        params = f"target_len={target_len}"
        df_json = df.to_json(
            orient="split",
            date_format="iso",
            date_unit="ns",
            default_handler=str,
        )
        if df_json is None:
            df_json = ""
        df_signature = df_json.encode("utf-8")
        payload = params.encode("utf-8") + b"|" + df_signature
        stem = hashlib.sha256(payload).hexdigest()
        return cache_dir / f"{stem}.pkl"
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load pre-trained OPED model from a state_dict.

        Args:
            model_path: Path to an OPED state_dict file (``*_weights.pt``) or a
                directory containing one. When ``None``, the weights bundled in
                ``vendor/models`` are used.

        Notes:
            Only state_dict files are supported. OPED's legacy full-pickle
            artifacts are version-fragile and are explicitly rejected; convert
            them once with ``app.models.convert_oped_weights``.
        """
        from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderModelOrder3

        weights_path = self._resolve_weights_path(model_path)
        self.model_dir = os.path.dirname(weights_path)

        # Architecture must match the trained checkpoint. ntokens is a per-order
        # list ([4, 16, 64]); the embeddings are sized ntokens+1 to reserve the
        # padding index 0.
        self.model = TransformerEncoderModelOrder3(
            ntokens=self.MODEL_NTOKENS,
            embedding_size=64,
            hidden_size=[2048, 2048, 2048],
            hidden_size_fully=None,
            output_size=1,
            nhead=8,
            num_encoder_layers=[6, 6, 6],
            dropout=0.1,
            other_size=0
        )

        state_dict = self._load_state_dict(weights_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.is_trained = True
    
    # OPED native sequence columns (output of PE-DB's oped converter).
    OPED_REQUIRED_COLUMNS = {"Target(47bp)", "PBS", "RT"}

    # model handles data loading and batching
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Prepare data for OPED model input by tokenizing native OPED sequences.

        Args:
            df: Native OPED dataframe with columns ``Target(47bp)``, ``PBS``,
                ``RT`` (and optionally ``Efficiency``). Fetch model-format data
                from PE-DB (``GET /api/filter?...&format=oped``). Train/test
                assignments come from PE-DB ``split`` columns, not this method.
            **kwargs:
                - target_len (int): target sequence length (default 47)
                - use_cache (bool): if True, reuse encoded data from cache

        Returns:
            Encoded OPED dataframe
        """
        target_len = int(kwargs.get("target_len", 47))
        use_cache = bool(kwargs.get("use_cache", False))

        cache_path = self._build_cache_path(
            df=df,
            target_len=target_len,
        )
        if use_cache and cache_path.exists():
            cached_obj = pd.read_pickle(cache_path)
            if not isinstance(cached_obj, pd.DataFrame):
                raise ValueError(
                    f"Invalid OPED cache format at {cache_path}. Expected DataFrame."
                )
            encoded_full = cached_obj
        else:
            if not self.OPED_REQUIRED_COLUMNS.issubset(df.columns):
                missing = sorted(self.OPED_REQUIRED_COLUMNS.difference(df.columns))
                raise ValueError(
                    "OPED expects native sequence columns; missing: "
                    f"{missing}. Fetch model-format data from PE-DB "
                    "(GET /api/filter?...&format=oped)."
                )
            if df.empty:
                raise ValueError("No valid rows after OPED data preparation.")

            # Tokenize the native OPED sequences. _to_oped_numeric_df returns a
            # fresh RangeIndex frame, so all attached columns use positional
            # alignment via to_numpy().
            encoded_full = self._to_oped_numeric_df(df)
            if "Efficiency" in df.columns:
                encoded_full["Efficiency"] = (
                    pd.to_numeric(df["Efficiency"], errors="coerce").fillna(0.0).to_numpy()
                )
            else:
                encoded_full["Efficiency"] = np.zeros(len(df), dtype=float)
            for meta_col in ("split", "split_source", "original_fold"):
                if meta_col in df.columns:
                    encoded_full[meta_col] = df[meta_col].to_numpy()
            if use_cache:
                pd.to_pickle(encoded_full, cache_path)

        return encoded_full
    
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
        
        from oped.pegRNA_PredictingCodes.evaluate_model import transformer_predictor_order3
        
        outputs, _ = transformer_predictor_order3(
            transformer=self.model,
            X_test=data,
            batch_size_test=batch_size,
            device=self.device
        )
        
        return outputs.tolist() if isinstance(outputs, np.ndarray) else outputs

    @staticmethod
    def _build_model_from_hparams(hparams: Dict[str, Any]) -> torch.nn.Module:
        from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderModelOrder3

        ntokens_value = hparams.get("ntokens", hparams.get("ntoken", [4, 16, 64]))
        if isinstance(ntokens_value, int):
            ntokens_value = [ntokens_value, 16, 64]
        model = TransformerEncoderModelOrder3(
            ntokens=ntokens_value,
            embedding_size=int(hparams.get("embedding_size", 64)),
            hidden_size=hparams.get("hidden_size", [2048, 2048, 2048]),
            hidden_size_fully=hparams.get("hidden_size_fully", None),
            output_size=int(hparams.get("output_size", 1)),
            nhead=int(hparams.get("nhead", 8)),
            num_encoder_layers=hparams.get("num_encoder_layers", [6, 6, 6]),
            dropout=float(hparams.get("drop_out", hparams.get("dropout", 0.1))),
            other_size=int(hparams.get("other_size", 0)),
        )
        return model

    @staticmethod
    def _batch_inputs_from_encoded_df(batch_df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, ...]:
        return (
            torch.tensor(list(batch_df["Target"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["PBS"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["RT"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["Target_o2"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["PBS_o2"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["RT_o2"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["Target_o3"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["PBS_o3"]), device=device, dtype=torch.long),
            torch.tensor(list(batch_df["RT_o3"]), device=device, dtype=torch.long),
        )

    def _predict_encoded_df(
        self,
        model: torch.nn.Module,
        encoded_df: pd.DataFrame,
        batch_size: int = 1024,
    ) -> np.ndarray:
        model.eval()
        outputs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(encoded_df), batch_size):
                xb = encoded_df.iloc[start : start + batch_size, :]
                if xb.empty:
                    continue
                inputs = self._batch_inputs_from_encoded_df(xb, self.device)
                pred, _ = model(inputs)
                outputs.extend(pred.squeeze(-1).detach().cpu().numpy().tolist())
        return np.asarray(outputs, dtype=float)

    def _run_training_loop(
        self,
        model: torch.nn.Module,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hparams: Dict[str, Any],
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        batch_size = int(hparams.get("batch_size", 128))
        num_epochs = int(hparams.get("epoch_num", hparams.get("epochs", 100)))
        grad_clip = float(hparams.get("grad_clip", 1.0))
        early_stopping_patience = int(hparams.get("early_stopping_patience", 10))
        early_stopping_delta = float(hparams.get("early_stopping_min_delta", 0.0))
        train_loader = DataLoader(
            _OPEDEncodedDataset(train_df),
            batch_size=batch_size,
            shuffle=bool(hparams.get("reshuffle_each_epoch", True)),
            drop_last=False,
        )
        val_loader = DataLoader(
            _OPEDEncodedDataset(val_df),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        lightning_module = _OPEDLightningRegressor(model=model, hparams=hparams)
        metrics = fit_lightning_module(
            lightning_module,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            config=LightningTrainerConfig(
                max_epochs=num_epochs,
                grad_clip=grad_clip,
                patience=early_stopping_patience,
                min_delta=early_stopping_delta,
                enable_progress_bar=bool(hparams.get("progress_bar", False)),
                log_every_n_steps=int(hparams.get("log_every_n_steps", 25)),
            ),
        )
        model = lightning_module.model
        return model, {
            "history": metrics["history"],
            "best_val_loss": float(metrics["best_val_loss"]),
            "best_epoch": int(metrics["best_epoch"]),
            "n_epochs_ran": int(metrics["n_epochs_ran"]),
        }

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
        # Assume caller provides OPED-encoded data (output of prepare_data).
        if "Efficiency" not in train_data.columns:
            raise ValueError("train_data must include 'Efficiency' column.")
        if val_data is not None and "Efficiency" not in val_data.columns:
            raise ValueError("val_data must include 'Efficiency' column.")

        default_params: Dict[str, Any] = {
            "ntokens": [4, 16, 64],
            "embedding_size": 64,
            "hidden_size": [2048, 2048, 2048],
            "hidden_size_fully": None,
            "output_size": 1,
            "nhead": 8,
            "num_encoder_layers": [6, 6, 6],
            "drop_out": 0.1,
            "epoch_num": 100,
            "batch_size": 128,
            "lr": 3e-4,
            "weight_decay": 0.0,
            "other_size": 0,
            # Shared training controls
            "grad_clip": 1.0,
            "reshuffle_each_epoch": True,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.0,
            "scheduler": "step",
            "scheduler_kwargs": {"step_size": 10, "gamma": 0.95},
        }
        if hyperparameters:
            default_params.update(hyperparameters)

        def apply_freezing_if_needed(model_obj: torch.nn.Module) -> None:
            if not freezing:
                return
            for param in model_obj.parameters():
                param.requires_grad = False
            fc_layers = getattr(model_obj, "fully_connected_layers", None)
            if isinstance(fc_layers, torch.nn.Module):
                for param in fc_layers.parameters():
                    param.requires_grad = True

        source_df = train_data.copy().reset_index(drop=True)
        cv_reports: List[Dict[str, Any]] = []

        if val_data is None and has_assigned_cv_folds(source_df):
            for fold_idx, (fold_label, fold_train, fold_val) in enumerate(
                iter_assigned_cv_folds(source_df)
            ):
                fold_model = self._build_model_from_hparams(default_params).to(self.device)
                apply_freezing_if_needed(fold_model)
                fold_model, fold_metrics = self._run_training_loop(
                    fold_model,
                    fold_train,
                    fold_val,
                    default_params,
                )
                fold_pred = self._predict_encoded_df(
                    fold_model,
                    fold_val.drop(columns=["Efficiency"], errors="ignore"),
                    batch_size=int(default_params.get("batch_size", 128)),
                )
                fold_true = fold_val["Efficiency"].astype(float).to_numpy()
                fold_corr = pearson_spearman(fold_true.tolist(), fold_pred.tolist())
                cv_reports.append(
                    {
                        "fold": fold_idx,
                        "fold_label": fold_label,
                        "n_train": int(len(fold_train)),
                        "n_val": int(len(fold_val)),
                        "best_epoch": int(fold_metrics["best_epoch"]),
                        "best_val_loss": float(fold_metrics["best_val_loss"]),
                        "val_pearson": float(fold_corr["pearson"]),
                        "val_spearman": float(fold_corr["spearman"]),
                    }
                )

        final_train, final_val = resolve_train_val_from_splits(source_df, val_data)

        model = self._build_model_from_hparams(default_params).to(self.device)
        apply_freezing_if_needed(model)
        model, train_metrics = self._run_training_loop(model, final_train, final_val, default_params)
        val_pred = self._predict_encoded_df(
            model,
            final_val.drop(columns=["Efficiency"]),
            batch_size=int(default_params.get("batch_size", 128)),
        )
        val_true = final_val["Efficiency"].astype(float).to_numpy()
        val_corr = pearson_spearman(val_true.tolist(), val_pred.tolist())

        self.model = model
        self.is_trained = True

        result: Dict[str, Any] = {
            "status": "success",
            "hyperparameters": default_params,
            "best_epoch": int(train_metrics["best_epoch"]),
            "best_val_loss": float(train_metrics["best_val_loss"]),
            "val_pearson": float(val_corr["pearson"]),
            "val_spearman": float(val_corr["spearman"]),
            "n_train_final": int(len(final_train)),
            "n_val_final": int(len(final_val)),
        }
        if cv_reports:
            pearsons = [r["val_pearson"] for r in cv_reports if not np.isnan(r["val_pearson"])]
            spearmans = [r["val_spearman"] for r in cv_reports if not np.isnan(r["val_spearman"])]
            result["cross_validation"] = {
                "folds": cv_reports,
                "mean_val_pearson": float(np.mean(pearsons)) if pearsons else float("nan"),
                "mean_val_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
            }
        return result
    
    def evaluate(self, test_data: pd.DataFrame, weights: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate OPED model

        Args:
            test_data: Test DataFrame with features and 'Efficiency' label
            weights: Optional name of a bundled pre-trained weight set to load
                before evaluating (see :meth:`list_available_weights`). When
                ``None``, the currently trained/loaded model is used.

        Returns:
            Dictionary with evaluation metrics
        """
        if weights is not None:
            self.load_weights_by_name(weights)

        if not self.is_trained:
            raise ValueError(
                "Model not loaded. Pass `weights=<name>` (see list_available_weights()), "
                "or call load_model()/train() first."
            )
        
        from oped.pegRNA_PredictingCodes.evaluate_model import evaluate_transformer_order3
        
        # Assume caller provides OPED-encoded data (e.g. output of prepare_data).
        X_test = test_data.copy()
        y_test = X_test["Efficiency"].astype(float).reset_index(drop=True)
        X_test = X_test.drop(columns=["Efficiency"])
        
        # Evaluate
        results, _, _ = evaluate_transformer_order3(
            transformer=self.model,
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
        torch.save(self.model.state_dict(), model_path)
        
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

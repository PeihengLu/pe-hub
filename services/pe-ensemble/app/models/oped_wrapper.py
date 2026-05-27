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
from pe_common.model_interface import BasePEModel
from pe_common.constants import DEVICE, DATA_ROOT
from pe_common.sequence_utils import (
    reverse_complement,
    sanitize_dna_sequence,
)
from pe_common.data_utils import build_test_mask_from_group_id
from pe_common.training import (
    build_lr_scheduler,
    build_group_kfold_indices,
    fit_lightning_module,
    LightningTrainerConfig,
    pearson_spearman,
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
    
    def __init__(self, device: torch.device = torch.device(DEVICE)):
        super().__init__('OPED', device)
        self.model_dir = None

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


    def _standardized_to_oped_sequence_df(
        self,
        df: pd.DataFrame,
        target_len: int = 47,
        protospacer_upstream_bases: int = 4,
    ) -> pd.DataFrame:
        seq_records = []
        efficiency = df["editing_efficiency"].astype(float).to_numpy()
        wt_series = df["wt_sequence"].astype(str).str.upper().to_numpy()
        mut_series = df["mut_sequence"].astype(str).str.upper().to_numpy()
        pbs_l = df["pbs_location_l"].astype(int).to_numpy()
        pbs_r = df["pbs_location_r"].astype(int).to_numpy()
        rtt_l = df["rtt_location_l"].astype(int).to_numpy()
        rtt_r = df["rtt_location_r"].astype(int).to_numpy()
        prot_l = df["protospacer_location_l"].astype(int).to_numpy()
        index_values = df.index.to_numpy()

        for row_pos, (idx, wt, mut, pbs_l_i, pbs_r_i, rtt_l_i, rtt_r_i, prot_l_i) in enumerate(
            zip(
            index_values, wt_series, mut_series, pbs_l, pbs_r, rtt_l, rtt_r, prot_l
            )
        ):

            # Use WT base, and fill masked/alignment chars from MUT.
            ref_chars = []
            for i in range(min(len(wt), len(mut))):
                wt_base = wt[i]
                mut_base = mut[i]
                if wt_base in {"A", "C", "G", "T"}:
                    ref_chars.append(wt_base)
                elif mut_base in {"A", "C", "G", "T"}:
                    ref_chars.append(mut_base)
                else:
                    ref_chars.append("A")
            if len(wt) > len(mut):
                for base in wt[len(mut) :]:
                    ref_chars.append(base if base in {"A", "C", "G", "T"} else "A")
            elif len(mut) > len(wt):
                for base in mut[len(wt) :]:
                    ref_chars.append(base if base in {"A", "C", "G", "T"} else "A")
            ref_seq = "".join(ref_chars)

            # Build OPED's Target(47bp) as a protospacer-anchored window:
            # keep a fixed number of upstream bases before protospacer start.
            target_start = max(0, int(prot_l_i) - int(protospacer_upstream_bases))
            target_end = target_start + target_len
            if target_end > len(ref_seq):
                target_start = max(0, len(ref_seq) - target_len)
                target_end = len(ref_seq)

            target = ref_seq[target_start:target_end]
            pbs_l_i = max(0, int(pbs_l_i))
            pbs_r_i = min(len(ref_seq), int(pbs_r_i))
            rtt_l_i = max(0, int(rtt_l_i))
            rtt_r_i = min(len(ref_seq), int(rtt_r_i))
            pbs = ref_seq[pbs_l_i:pbs_r_i]
            rt = ref_seq[rtt_l_i:rtt_r_i]

            target = sanitize_dna_sequence(target)
            pbs = sanitize_dna_sequence(pbs)
            rt = sanitize_dna_sequence(rt)

            if len(target) < target_len:
                target = target + ("A" * (target_len - len(target)))

            record = {"Target(47bp)": target, "PBS": pbs, "RT": rt, "_source_index": idx}
            record["Efficiency"] = float(efficiency[row_pos])
            seq_records.append(record)

        if not seq_records:
            return pd.DataFrame(
                columns=pd.Index(["Target(47bp)", "PBS", "RT", "Efficiency"])
            )
        out_df = pd.DataFrame(seq_records).set_index("_source_index")
        return out_df

    def _split_train_test_by_fold_or_group(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        test_fold_value: int = -1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fold_col = "original_fold"

        test_mask = df[fold_col] == test_fold_value
        if bool(test_mask.any()) and bool((~test_mask).any()):
            return df.loc[~test_mask].copy(), df.loc[test_mask].copy()

        # fallback split by standardized group id when fold split is unavailable
        group_col = "group_id"
        rng = np.random.default_rng(random_state)
        test_mask = build_test_mask_from_group_id(
            group_series=pd.Series(df[group_col], copy=False),
            test_size=test_size,
            random_state=random_state,
        )

        if not bool((~test_mask).any()) or not bool(test_mask.any()):
            row_indices = np.arange(len(df))
            rng.shuffle(row_indices)
            n_test_rows = max(1, int(np.ceil(len(df) * test_size)))
            test_idx = set(row_indices[:n_test_rows].tolist())
            test_mask = pd.Series([i in test_idx for i in range(len(df))], index=df.index)

        return df.loc[~test_mask].copy(), df.loc[test_mask].copy()

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
    
    def load_model(self, model_path: str) -> None:
        """
        Load pre-trained OPED model
        
        Args:
            model_path: Path to the saved OPED model file (.pt or .pkl)
        """
        from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderModelOrder3
        
        self.model_dir = os.path.dirname(model_path)
        
        # Initialize model architecture (these should match the trained model)
        # Default parameters - adjust based on your trained model
        self.model = TransformerEncoderModelOrder3(
            ntokens=4,
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
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.is_trained = True
    
    # model handles data loading and batching
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Prepare data for OPED model input.
        
        Args:
            df: Standardized PE dataframe (canonical standardized columns).
            **kwargs:
                - split_data (bool): when True, return (train_df, test_df)
                - test_size (float): group split ratio when fold is unavailable
                - random_state (int): random seed for group split
                - holdout_fold_value (int): fold value treated as test split
                - use_cache (bool): if True, reuse converted data from cache
            
        Returns:
            Encoded OPED dataframe, or (train_df, test_df) if split_data=True

        Examples:
            >>> prepared = model.prepare_data(df_standardized)
            >>> train_df, test_df = model.prepare_data(
            ...     df_standardized, split_data=True, test_size=0.2, random_state=42
            ... )
        """
        split_data = kwargs.get("split_data", False)
        test_size = float(kwargs.get("test_size", 0.2))
        random_state = int(kwargs.get("random_state", 42))
        holdout_fold_value = int(kwargs.get("holdout_fold_value", -1))
        target_len = int(kwargs.get("target_len", 47))
        protospacer_upstream_bases = int(kwargs.get("protospacer_upstream_bases", 4))
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
            sequence_df = self._standardized_to_oped_sequence_df(
                df,
                target_len=target_len,
                protospacer_upstream_bases=protospacer_upstream_bases,
            )
            if sequence_df.empty:
                raise ValueError("No valid rows after OPED data preparation.")

            encoded_full = self._to_oped_numeric_df(sequence_df)
            encoded_full["Efficiency"] = sequence_df["Efficiency"].astype(float).to_numpy()
            # Preserve split metadata in cache so we can split later without reconversion.
            metadata_df = df.reindex(sequence_df.index)
            # Keep missing split metadata as NaN (e.g. datasets with unknown original split).
            encoded_full["original_fold"] = pd.to_numeric(
                metadata_df.get("original_fold", pd.Series(np.nan, index=metadata_df.index)),
                errors="coerce",
            )
            encoded_full["group_id"] = pd.to_numeric(
                metadata_df.get("group_id", pd.Series(np.nan, index=metadata_df.index)),
                errors="coerce",
            )
            if use_cache:
                pd.to_pickle(encoded_full, cache_path)

        if split_data:
            train_full, test_full = self._split_train_test_by_fold_or_group(
                df=encoded_full,
                test_size=test_size,
                random_state=random_state,
                test_fold_value=holdout_fold_value,
            )
            train_encoded = train_full.drop(columns=["original_fold", "group_id"])
            test_encoded = test_full.drop(columns=["original_fold", "group_id"])
            return train_encoded, test_encoded

        return encoded_full.drop(columns=["original_fold", "group_id"])
    
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
            # CV controls (applied when val_data is None and cv_folds > 1)
            "cv_folds": 1,
            "cv_group_col": "group_id",
            "cv_random_state": 42,
            # Holdout split controls for final fit when val_data is absent
            "val_fraction": 0.1,
            "val_random_state": 42,
        }
        if hyperparameters:
            default_params.update(hyperparameters)

        # Optional layer freezing for transfer-learning-style finetuning.
        def apply_freezing_if_needed(model_obj: torch.nn.Module) -> None:
            if not freezing:
                return
            for param in model_obj.parameters():
                param.requires_grad = False
            fc_layers = getattr(model_obj, "fully_connected_layers", None)
            if isinstance(fc_layers, torch.nn.Module):
                for param in fc_layers.parameters():
                    param.requires_grad = True

        train_df = train_data.copy().reset_index(drop=True)
        cv_reports: List[Dict[str, Any]] = []
        cv_folds = int(default_params.get("cv_folds", 1))

        if val_data is None and cv_folds > 1:
            folds = build_group_kfold_indices(
                train_df,
                n_splits=cv_folds,
                group_col=str(default_params.get("cv_group_col", "group_id")),
                random_state=int(default_params.get("cv_random_state", 42)),
            )
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
                fold_val = train_df.iloc[va_idx].reset_index(drop=True)
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
                    fold_val.drop(columns=["Efficiency"]),
                    batch_size=int(default_params.get("batch_size", 128)),
                )
                fold_true = fold_val["Efficiency"].astype(float).to_numpy()
                fold_corr = pearson_spearman(fold_true.tolist(), fold_pred.tolist())
                cv_reports.append(
                    {
                        "fold": fold_idx,
                        "n_train": int(len(fold_train)),
                        "n_val": int(len(fold_val)),
                        "best_epoch": int(fold_metrics["best_epoch"]),
                        "best_val_loss": float(fold_metrics["best_val_loss"]),
                        "val_pearson": float(fold_corr["pearson"]),
                        "val_spearman": float(fold_corr["spearman"]),
                    }
                )

        if val_data is not None:
            final_train = train_df
            final_val = val_data.copy().reset_index(drop=True)
        else:
            val_fraction = float(default_params.get("val_fraction", 0.1))
            val_fraction = max(0.05, min(0.5, val_fraction))
            n_rows = len(train_df)
            if n_rows < 2:
                raise ValueError("Need at least 2 rows to train OPED model.")
            n_val = max(1, int(np.ceil(n_rows * val_fraction)))
            rng = np.random.default_rng(int(default_params.get("val_random_state", 42)))
            all_idx = np.arange(n_rows)
            rng.shuffle(all_idx)
            val_idx = all_idx[:n_val]
            train_idx = all_idx[n_val:]
            if len(train_idx) == 0:
                train_idx = val_idx[:1]
                val_idx = all_idx[1:]
            final_train = train_df.iloc[train_idx].reset_index(drop=True)
            final_val = train_df.iloc[val_idx].reset_index(drop=True)

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

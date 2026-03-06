import sys
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch
import numpy as np

from .vendor_path import resolve_vendor_models_path
from pe_common.model_interface import BasePEModel
from pe_common.constants import DEVICE, DATA_ROOT
from pe_common.sequence_utils import (
    reverse_complement,
    sanitize_dna_sequence,
)

# Ensure that the OPED model code directory is in sys.path
_vendor_root = resolve_vendor_models_path()
sys.path.insert(0, str(_vendor_root))

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
            return pd.DataFrame(
                columns=[
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
        self, df: pd.DataFrame, target_len: int = 47
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

        for idx, wt, mut, pbs_l_i, pbs_r_i, rtt_l_i, rtt_r_i, prot_l_i in zip(
            index_values, wt_series, mut_series, pbs_l, pbs_r, rtt_l, rtt_r, prot_l
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

            # Infer spacer prefix from standardized protospacer_location_l.
            target_start = max(0, prot_l_i - int(prot_l_i))
            target_end = target_start + target_len
            if target_end > len(ref_seq):
                target_start = max(0, len(ref_seq) - target_len)
                target_end = len(ref_seq)

            target = ref_seq[target_start:target_end]
            pbs = ref_seq[pbs_l_i:pbs_r_i]
            rt = ref_seq[rtt_l_i:rtt_r_i]

            target = sanitize_dna_sequence(target)
            pbs = sanitize_dna_sequence(pbs)
            rt = sanitize_dna_sequence(rt)

            if len(target) < target_len:
                target = target + ("A" * (target_len - len(target)))

            record = {"Target(47bp)": target, "PBS": pbs, "RT": rt, "_source_index": idx}
            record["Efficiency"] = efficiency[idx]
            seq_records.append(record)

        if not seq_records:
            return pd.DataFrame(columns=["Target(47bp)", "PBS", "RT", "Efficiency"])
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
        group_values = df[group_col].dropna().unique().tolist()

        rng.shuffle(group_values)
        n_test_groups = max(1, int(np.ceil(len(group_values) * test_size)))
        test_groups = group_values[:n_test_groups]
        test_mask = df[group_col].isin(test_groups)

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
            sequence_df = self._standardized_to_oped_sequence_df(df, target_len=target_len)
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
        from oped.pegRNA_PredictingCodes.train_model import train_and_test_transformer_order3
        
        # Assume caller provides OPED-encoded data (e.g. output of prepare_data).
        X_train = train_data.copy()
        X_train_y = X_train["Efficiency"].astype(float).reset_index(drop=True)
        X_train = X_train.drop(columns=["Efficiency"])
        
        if val_data is not None:
            X_val = val_data.copy()
            X_val_y = X_val["Efficiency"].astype(float).reset_index(drop=True)
            X_val = X_val.drop(columns=["Efficiency"])
        else:
            # Use training data as validation if not provided
            X_val, X_val_y = X_train, X_train_y
        
        # Default hyperparameters for OPED transformer
        default_params = {
            'ntoken': 4,
            'embedding_size': 64,
            'hidden_size': [2048, 2048, 2048],
            'hidden_size_fully': None,
            'output_size': 1,
            'nhead': 8,
            'num_encoder_layers': [6, 6, 6],
            'drop_out': 0.1,
            'epoch_num': 100,
            'batch_size': 128,
            'lr': 0.001,
            'weight_decay': 0.0,
            'device': self.device,
            'best_epoch': True,
            'transfer': False,
            'freezing': freezing,
            'other_size': 0
        }
        
        if hyperparameters:
            default_params.update(hyperparameters)
        
        # Train model
        trained_model = train_and_test_transformer_order3(
            X_train=X_train,
            X_test=X_val,
            y_train=X_train_y,
            y_test=X_val_y,
            hyperparameters=default_params,
            transformer=None  # Train from scratch
        )
        
        self.model = trained_model
        self.is_trained = True
        
        # Evaluate on validation set
        from oped.pegRNA_PredictingCodes.evaluate_model import evaluate_transformer_order3
        
        results, _, _ = evaluate_transformer_order3(
            transformer=self.model,
            X_train=X_val,
            y_train=X_val_y,
            batch_size_test=1024,
            device=self.device,
            verbose=True
        )
        
        return {
            'status': 'success',
            'hyperparameters': default_params,
            'val_pearson': results['pearson'][0],
            'val_spearman': results['spearman'][0]
        }
    
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

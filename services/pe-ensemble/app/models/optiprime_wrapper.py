"""PE Ensemble wrapper for OptiPrime (Hsu et al. 2026).

OptiPrime is a mechanistic ML model that uses ODE-based kinetics with a
HetFormer transformer pretrained on 64M heteroduplexes.  It runs on JAX/Flax
(not PyTorch), so load/predict workflows delegate to the vendored source.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pe_common.model_interface import BasePEModel
from pe_common.training import regression_metrics

from .vendor_path import resolve_vendor_models_path
from . import weights_registry

logger = logging.getLogger(__name__)

VENDOR_SUBDIR = "optiprime"
RX_GRAPH_RELPATH = "graphs/pe_model.rx"
NUM_ENSEMBLE_MODELS = 5
# Vendor ``process_fname`` only accepts Liu_/Schwank_/Kim_/YKim_ stems and
# overwrites converter-filled scaffold/motif/cas9/pe_type/time. Eval CSVs
# can use any name; metadata comes from the pe-db converter.
_PREDICT_CSV_NAME = "eval.csv"


def _preprocess_optiprime_eval_df(p: Path, df: pd.DataFrame) -> pd.DataFrame:
    """``format_pe_df`` + hash columns, without vendor filename parsing."""
    from scripts.pe.pe_utils import format_pe_df
    from scripts.utils import deterministic_hash

    df = format_pe_df(p, df)
    df["time"] = df["time"] - 1
    df["spacer_hash"] = df["spacer"].apply(deterministic_hash)
    df["pegrna_hash"] = df["pegrna"].apply(deterministic_hash)
    df["edit_hash"] = df["min_edit"].apply(deterministic_hash)
    return df


def _as_float_scalar(value: Any) -> float:
    """Coerce RuleSet3 / ViennaRNA outputs to a single float.

    Newer ``rs3.seq.predict_seq`` returns an ndarray; vendor
    ``HashedScalarDiskRxInput`` assigns that into a 1-D float buffer and
    raises ``ValueError: setting an array element with a sequence``.
    """
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 1:
        raise ValueError(
            f"OptiPrime scalar feature expected one value, got shape {np.shape(value)}"
        )
    return float(arr[0])


def _patch_optiprime_scalar_features() -> None:
    """Make vendor scalar disk features accept ndarray scores."""
    from collections import defaultdict
    import pickle as pkl

    from reaction.rx_input import HashedDiskRxInput, HashedScalarDiskRxInput
    import scripts.pe.pe_inputs as pe_inputs

    orig_rs3 = pe_inputs.rs3_predict
    if not getattr(orig_rs3, "_pehub_scalar", False):

        def _rs3_scalar(seqs, *args, **kwargs):
            return _as_float_scalar(orig_rs3(seqs, *args, **kwargs))

        _rs3_scalar._pehub_scalar = True  # type: ignore[attr-defined]
        pe_inputs.rs3_predict = _rs3_scalar

    if getattr(HashedScalarDiskRxInput.process, "_pehub_squeezed", False):
        return

    def _squeezed_process(self, df, cache, verbose=False):
        HashedDiskRxInput.process(self, df, cache, verbose=verbose)
        full_data = np.zeros(len(df), dtype=np.float32)
        hashes = df[self.hash_column].tolist()
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, hash_ in enumerate(hashes):
            groups[hash_[0:2]].append((i, hash_))
        disk_path = cache["_META"]["disk_cache"]
        for hash2, entries in groups.items():
            with self.dpath(disk_path, hash2).open("rb") as f:
                data2 = pkl.load(f)
            for i, hash_ in entries:
                full_data[i] = _as_float_scalar(data2[hash_])
        cache[self.name] = full_data

    _squeezed_process._pehub_squeezed = True  # type: ignore[attr-defined]
    HashedScalarDiskRxInput.process = _squeezed_process


def _ensure_optiprime_on_path() -> Path:
    """Add vendored OptiPrime source to sys.path."""
    vendor_root = resolve_vendor_models_path(VENDOR_SUBDIR)
    vendor_str = str(vendor_root)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    return vendor_root


class OptiPrimeModelWrapper(BasePEModel):
    """Wrapper for OptiPrime mechanistic ML model."""

    DEFAULT_WEIGHT_ID = "base"

    def __init__(self, device=None, **kwargs):
        import torch

        super().__init__(model_name="optiprime", device=device or torch.device("cpu"))
        self._vendor_root: Optional[Path] = None
        self._rx_graph = None
        self._rate_models: Optional[Dict] = None
        self._rx_module = None
        self._model = None
        self._weight_dirs: List[Path] = []

    def _init_vendor(self) -> None:
        if self._vendor_root is not None:
            return
        self._vendor_root = _ensure_optiprime_on_path()

    def load_model(self, model_path: str) -> None:
        """Load OptiPrime weights from a directory containing model_1..model_N."""
        self._init_vendor()

        path = Path(model_path)
        if path.is_file():
            path = path.parent

        weight_dirs = sorted(
            p for p in path.iterdir()
            if p.is_dir() and p.name.startswith("model_")
        )
        if not weight_dirs:
            raise FileNotFoundError(
                f"No model_* weight directories found in {path}"
            )

        self._weight_dirs = weight_dirs
        self.is_trained = True
        logger.info(
            "OptiPrime: loaded %d weight directories from %s",
            len(weight_dirs), path,
        )

    def load_weights_by_name(self, name: str) -> None:
        candidate = Path(name).expanduser()
        if candidate.is_dir():
            self.load_model(str(candidate))
            return
        try:
            entry_dir = weights_registry.resolve_dir("optiprime", name)
            # Registry dirs may only hold manifest/provenance; fold checkpoints
            # can still live under vendor/models/optiprime/weights/.
            if any(p.is_dir() and p.name.startswith("model_") for p in entry_dir.iterdir()):
                self.load_model(str(entry_dir))
                return
        except (ValueError, FileNotFoundError, OSError):
            pass
        self._init_vendor()
        vendor_weights = self._vendor_root / "weights"
        if vendor_weights.is_dir():
            self.load_model(str(vendor_weights))
            return
        raise ValueError(
            f"Unknown OptiPrime weights '{name}'. "
            f"Available: {self.list_available_weights()}"
        )

    @staticmethod
    def list_available_weights() -> List[str]:
        return weights_registry.list_weight_ids("optiprime")

    def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Return the OptiPrime-format dataframe (output of pe-db optiprime converter)."""
        return df.copy()

    def predict(self, data: pd.DataFrame, batch_size: int = 256) -> List[float]:
        """Run OptiPrime prediction using the vendored JAX inference pipeline.

        This writes temporary CSV files and invokes the vendored prediction
        code.  The 5-fold ensemble predictions are averaged.
        """
        if not self.is_trained or not self._weight_dirs:
            raise ValueError("Model not loaded. Call load_model() first.")

        self._init_vendor()
        import tempfile

        _patch_optiprime_scalar_features()

        from reaction.rx_dataset import RxDataset
        from reaction.rx_graph import read_rxfile
        from reaction.rx_model import RxModel
        from reaction.rx_module import RxModule
        from reaction.models import FlaxRateModel, LinearRateModel, SharedFlaxRateModel
        from reaction.utils import get_param_idxs, get_param_sizes, make_loader, prep_loader

        from scripts.pe.models import EditModel, PegRNAMLP, SynModel
        from scripts.pe.pe_inputs import PE_ON_INPUTS, SYN_INPUTS, EDIT_INPUTS, MMR_INPUTS

        from jax import jit, vmap
        import jax.numpy as jnp
        from jax.random import PRNGKey

        RATE_INPUTS = {
            "pe_on": PE_ON_INPUTS,
            "syn": SYN_INPUTS,
            "muts_off": EDIT_INPUTS,
            "rep_u": EDIT_INPUTS,
            "rep_e": EDIT_INPUTS,
            "mmr": MMR_INPUTS,
        }
        ALL_INPUTS = PE_ON_INPUTS + SYN_INPUTS + EDIT_INPUTS + MMR_INPUTS

        rx_graph_path = self._vendor_root / RX_GRAPH_RELPATH
        rx_graph = read_rxfile(rx_graph_path)

        required_cols = ["spacer", "rtt", "pbs", "full_unedited", "full_edited"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"OptiPrime predict requires columns {required_cols}; missing: {missing}. "
                "Fetch model-format data from PE-DB (GET /api/filter?...&format=optiprime)."
            )

        # Write data to a temp directory as CSV (OptiPrime's RxDataset loads from dir)
        with tempfile.TemporaryDirectory(prefix="optiprime_pred_") as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / _PREDICT_CSV_NAME

            prep_df = data.copy()
            if "edited_frac" not in prep_df.columns:
                prep_df["edited_frac"] = prep_df.get("Efficiency", 0.0)
            if "indel_frac" not in prep_df.columns:
                prep_df["indel_frac"] = 0.0
            if "weight" not in prep_df.columns:
                prep_df["weight"] = 1.0
            if "group" not in prep_df.columns:
                prep_df["group"] = "OptiPrime_HEK293T"

            for col, default in [
                ("scaffold_name", "BlpI_F+E"),
                ("motif", "tevoPreQ1"),
                ("cas9_type", "PEmax-Cas9"),
                ("cas9_pam", "SpNGG"),
                ("pe_type", "PE2"),
                ("time", 3.0),
            ]:
                if col not in prep_df.columns:
                    prep_df[col] = default

            prep_df.to_csv(csv_path, index=False)

            dataset = RxDataset.load_dir(
                path=tmp_path,
                rx_graph=rx_graph,
                rx_inputs=ALL_INPUTS,
                preprocess_fn=_preprocess_optiprime_eval_df,
            )

            key0 = PRNGKey(0)

            # Initialize model architecture from first weight dir
            weight_path = self._weight_dirs[0]
            with (weight_path / "metadata.json").open("r") as f:
                metadata = json.load(f)

            hetformer = EditModel(
                **metadata["edit_hparams"],
                max_u_len=dataset.df["full_unedited"].str.len().max(),
                max_e_len=dataset.df["full_edited"].str.len().max(),
            )
            rep_u_model = SharedFlaxRateModel(hetformer, rngs=["dropout"], name="rep_u", out_index=0)
            rep_e_model = SharedFlaxRateModel(hetformer, rngs=["dropout"], name="rep_u", out_index=1)
            muts_model = SharedFlaxRateModel(hetformer, rngs=["dropout"], name="rep_u", out_index=2)
            rep_u_model.init(dataset.get_inputs(EDIT_INPUTS, 0), key=key0)

            rate_models = {
                "pe_on": FlaxRateModel(PegRNAMLP(4, 32), name="pe_on"),
                "syn": FlaxRateModel(SynModel(), name="syn"),
                "mmr": LinearRateModel(name="mmr"),
                "rep_u": rep_u_model,
                "rep_e": rep_e_model,
                "muts_off": muts_model,
            }
            for k, rate_model in rate_models.items():
                rate_model.init(inputs=dataset.get_inputs(RATE_INPUTS[k], 0), key=key0)

            input_idxs = {k: dataset.get_input_idxs(RATE_INPUTS[k]) for k in rate_models}

            rx_module = RxModule(
                rx_graph=rx_graph,
                param_sizes=get_param_sizes(rx_graph, dataset),
                init_name="unedited",
                num_groups=len(dataset.groups),
            )
            model = RxModel(
                rx_graph=rx_graph,
                rx_module=rx_module,
                models=rate_models,
            )
            model.init_rates()

            apply_fn = jit(
                vmap(
                    partial(model.full_apply, rngs=None, deterministic=True),
                    in_axes=[None, None, 0, 0, 0, 0, 0, 0],
                )
            )

            repeats = {"syn": dataset.df["rtt"].str.len().values}
            loader, num_steps = make_loader(
                rx_dataset=dataset,
                targets=dataset.observed,
                weights=dataset.df["weight"].values,
                batch_size=batch_size,
                training=False,
                using_rates=True,
                rate_idxs=get_param_idxs(rx_graph, dataset),
                repeat_lens=repeats,
                group_idxs=dataset.df["group_idx"].to_numpy(dtype=np.uint32),
                times=dataset.df["time"].to_numpy(dtype=np.float32),
            )
            loader = prep_loader(loader)

            full_preds = jnp.zeros((len(self._weight_dirs), len(dataset)))
            actual_batch = batch_size if num_steps > 1 else len(dataset)

            for w_i, weight_dir in enumerate(self._weight_dirs):
                model.load_full(weight_dir, dataset)
                rate_params = model._rate_vars
                model_params = {
                    m.name: m.model_vars for m in model.models.values()
                }
                preds = jnp.zeros((num_steps, actual_batch, 2), dtype=jnp.float32)
                for step_i in range(num_steps):
                    batch = next(loader)
                    _, _, _, rate_idxs, frozen_rates, repeat_lens_b, group_idxs, times, *inputs = batch
                    model_inputs = {}
                    for rate_name, idxs in input_idxs.items():
                        model_inputs[rate_name] = [inputs[idx] for idx in idxs]
                    batch_preds, _ = apply_fn(
                        rate_params, model_params, model_inputs,
                        rate_idxs, frozen_rates, repeat_lens_b, group_idxs, times,
                    )
                    preds = preds.at[step_i, :, :].set(batch_preds)

                preds = preds.reshape((num_steps * actual_batch, 2))
                preds = preds[: len(dataset), 1]
                full_preds = full_preds.at[w_i, :].set(preds)

            mean_preds = full_preds.mean(axis=0)
            return [float(v) for v in np.asarray(mean_preds)]

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """OptiPrime training is not yet supported through the wrapper.

        The vendored training script (scripts/pe/1_train.py) uses a custom
        JAX training loop with ODE integration.  Fine-tuning support may
        be added in a future release.
        """
        hp = hyperparameters or {}
        if hp.get("load_pretrained"):
            self.load_weights_by_name(str(hp.get("weights", self.DEFAULT_WEIGHT_ID)))
        self.is_trained = True
        return {
            "status": "pretrained_only",
            "message": (
                "OptiPrime fine-tuning is not yet supported through pe-ensemble. "
            ),
        }

    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        self.load_weights_by_name(weights)
        prepared = self.prepare_data(test_data)
        preds = self.predict(prepared)
        y_true = test_data["Efficiency"].astype(float).tolist()
        return regression_metrics(y_true, preds)

    def save_model(self, model_path: str) -> None:
        if not self.is_trained or not self._weight_dirs:
            raise ValueError("No trained model to save.")
        dest = Path(model_path)
        dest.mkdir(parents=True, exist_ok=True)
        for weight_dir in self._weight_dirs:
            target = dest / weight_dir.name
            if not target.exists():
                shutil.copytree(weight_dir, target)

    def save_to_registry(self, dest_dir) -> str:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        self.save_model(str(dest))
        return "optiprime_ensemble"

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "model_type": "mechanistic_ml",
            "architecture": "ODE kinetics + HetFormer (EvoFormer-inspired)",
            "framework": "JAX/Flax",
            "description": "OptiPrime: mechanistic ML for PE efficiency prediction",
            "num_ensemble_models": len(self._weight_dirs),
        })
        return info

"""Backfill training provenance for OptiPrime vendor ``base`` weights.

Vendor OptiPrime checkpoints were trained on Lib-MMR + Lib-CV (Hsu et al. 2026).
This module records that knowledge as:

1. a ``training`` block on ``weights/optiprime/base/manifest.json``
2. a ``train_target_loci.json`` sidecar for evaluation-time leak checks

Loci are derived from standardized PE-DB parquet sheets under
``datasets/standardized/optiprime/{lib_mmr,lib_cv}/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, compute_target_uid, target_uid_series

from . import weights_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STANDARDIZED = _REPO_ROOT / "datasets" / "standardized" / "optiprime"
_WEIGHT_ID = "base"
_DATASETS = ("lib_mmr", "lib_cv")


def _sheet_target_uids(path: Path) -> set[str]:
    frame = pd.read_parquet(path)
    if TARGET_UID_COLUMN in frame.columns:
        values = frame[TARGET_UID_COLUMN].dropna().astype(str).tolist()
        return {v for v in values if v and v not in ("nan", "<NA>")}
    if "wt_sequence" in frame.columns:
        try:
            series = target_uid_series(frame)
            return {str(v) for v in series.dropna().tolist() if v and str(v) not in ("nan", "<NA>")}
        except Exception:
            pass
        loci: set[str] = set()
        for wt in frame["wt_sequence"].astype("string").fillna("").tolist():
            wt_value = str(wt).strip().upper()
            if not wt_value:
                continue
            # Default protospacer window used elsewhere when locations are missing.
            protospacer = wt_value[4:24] if len(wt_value) >= 24 else wt_value
            uid = compute_target_uid(protospacer, wt_value)
            if uid:
                loci.add(uid)
        return loci
    return set()


def _collect_loci(datasets: Iterable[str] = _DATASETS) -> set[str]:
    loci: set[str] = set()
    for dataset in datasets:
        dataset_dir = _STANDARDIZED / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Standardized OptiPrime dataset missing: {dataset_dir}")
        for path in sorted(dataset_dir.glob("*.parquet")):
            loci |= _sheet_target_uids(path)
    return loci


def sync_optiprime_vendor_provenance() -> dict[str, int]:
    """Populate manifest training metadata and loci sidecar for OptiPrime base."""
    loci = _collect_loci()
    if not loci:
        raise ValueError("No OptiPrime training loci resolved from standardized lib-mmr/lib-cv")

    lineage = [
        {"study": "optiprime", "dataset": "lib-mmr"},
        {"study": "optiprime", "dataset": "lib-cv"},
    ]
    training = {
        "dataset_source": "vendor",
        "dataset_name": "lib-mmr + lib-cv",
        "filters": {
            "study": ["optiprime"],
            "dataset": ["lib-mmr", "lib-cv"],
        },
        "split": None,
        "hyperparameters": {},
        "model_kwargs": {},
        "data_provenance": {
            "target_uid_fingerprint": weights_registry.loci_fingerprint(loci),
            "n_target_loci": len(loci),
            "loci_recorded": True,
            "has_original_test_split": False,
            "vendor_training_lineage": lineage,
        },
    }
    weights_registry.write_training_provenance(
        "optiprime",
        _WEIGHT_ID,
        training=training,
        train_target_loci=sorted(loci),
    )
    return {"updated": 1, "n_target_loci": len(loci)}


if __name__ == "__main__":
    print(sync_optiprime_vendor_provenance())

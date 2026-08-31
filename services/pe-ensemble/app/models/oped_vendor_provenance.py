"""Backfill training provenance for registered OPED vendor weights.

Vendor OPED checkpoints were trained on DeepPE library train folds (Kim et al.),
not the author held-out test partition (``original_fold == -1``).

This module records that knowledge as:

1. a ``training`` block on the OPED weight ``manifest.json``
2. a ``train_target_loci.json`` sidecar for evaluation-time leak checks

Loci are derived from standardized PE-DB parquet sheets under
``datasets/standardized/deeppe/``, excluding author test rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, target_uid_series

from . import weights_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STANDARDIZED = _REPO_ROOT / "datasets" / "standardized" / "deeppe"
_WEIGHT_ID = "pegRNA_Model_Merged_saved.order3_decoder_weights"


def _is_author_train_fold(value: object) -> bool:
    """True only for labeled train folds (0, 1, …); False for test (-1) and NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if pd.isna(numeric):
        return False
    return numeric != -1.0


def _sheet_train_target_uids(path: Path) -> set[str]:
    frame = pd.read_parquet(path)
    if "original_fold" in frame.columns:
        frame = frame.loc[frame["original_fold"].map(_is_author_train_fold)].copy()
    if frame.empty:
        return set()
    if TARGET_UID_COLUMN in frame.columns:
        values = frame[TARGET_UID_COLUMN].dropna().astype(str).tolist()
        return {v for v in values if v and v not in ("nan", "<NA>")}
    try:
        series = target_uid_series(frame)
    except Exception:
        return set()
    return {str(v) for v in series.dropna().tolist() if v and str(v) not in ("nan", "<NA>")}


def _collect_train_loci() -> set[str]:
    root = _STANDARDIZED
    if not root.is_dir():
        raise FileNotFoundError(f"Standardized DeepPE dataset missing: {root}")
    loci: set[str] = set()
    for path in sorted(root.rglob("*.parquet")):
        loci |= _sheet_train_target_uids(path)
    return loci


def sync_oped_vendor_provenance() -> dict[str, int]:
    """Populate manifest training metadata and loci sidecar for OPED vendor weights."""
    loci = _collect_train_loci()
    if not loci:
        raise ValueError("No OPED training loci resolved from standardized DeepPE train folds")

    lineage = [{"study": "deeppe", "dataset": "deeppe (train folds)"}]
    training = {
        "dataset_source": "vendor",
        "dataset_name": "deeppe (train folds)",
        "filters": {
            "study": ["deeppe"],
            "dataset": [
                "deeppe-ht",
                "deeppe-type",
                "deeppe-position",
                "deeppe-endo",
            ],
        },
        "split": {
            "use_original_fold": True,
            "original_fold_test_value": -1.0,
        },
        "hyperparameters": {},
        "model_kwargs": {},
        "data_provenance": {
            "target_uid_fingerprint": weights_registry.loci_fingerprint(loci),
            "n_target_loci": len(loci),
            "loci_recorded": True,
            "has_original_test_split": True,
            "train_folds_only": True,
            "vendor_training_lineage": lineage,
        },
    }
    weights_registry.write_training_provenance(
        "oped",
        _WEIGHT_ID,
        training=training,
        train_target_loci=sorted(loci),
    )
    return {"updated": 1, "n_target_loci": len(loci)}


if __name__ == "__main__":
    print(sync_oped_vendor_provenance())

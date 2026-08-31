"""Backfill training provenance for OptiPrime vendor ``base`` weights.

Hsu et al. 2026 (Nature Biotechnology) jointly trained OptiPrime on Lib-MMR +
Lib-CV from this study plus PE efficiencies from prior publications (refs
54–56): PRIDICT library~1 (Mathis et al. 2023), DeepPrime ClinVar (Yu et al.
2023), and PRIDICT2 library-diverse (Mathis et al. 2025).  The vendored
``pe_datasets.process_fname`` loaders label these sources as ``Liu``,
``Schwank``, ``Kim``, and (for DeepPE) ``YKim``; the shipped ``base`` ensemble
was trained on the Liu + Schwank + Kim sources above.

This module records that knowledge as:

1. a ``training`` block on ``weights/optiprime/base/manifest.json``
2. a ``train_target_loci.json`` sidecar for evaluation-time leak checks

Loci are derived from standardized PE-DB parquet sheets.  DeepPrime ClinVar
rows with author ``original_fold == -1`` are excluded; all other sources use
every available row because PE-DB does not expose a vendor hold-out split for
PRIDICT1 library~1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, compute_target_uid, target_uid_series

from . import weights_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STANDARDIZED = _REPO_ROOT / "datasets" / "standardized"
_WEIGHT_ID = "base"


@dataclass(frozen=True)
class _DatasetSpec:
    study: str
    dataset: str
    lineage_dataset: str
    train_folds_only: bool = False


# OptiPrime Lib-MMR / Lib-CV (this study) plus external training corpora (refs 54–56).
_DATASET_SPECS: tuple[_DatasetSpec, ...] = (
    _DatasetSpec("optiprime", "lib_mmr", "lib-mmr"),
    _DatasetSpec("optiprime", "lib_cv", "lib-cv"),
    _DatasetSpec("pridict1", "library1", "library1"),
    _DatasetSpec("deepprime", "deepprime_clinvar", "deepprime-clinvar", train_folds_only=True),
    _DatasetSpec("pridict2", "library_diverse", "library-diverse"),
)


def _is_author_train_fold(value: object) -> bool:
    """True for labeled train folds (0, 1, …); False for author test (-1)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if pd.isna(numeric):
        return True
    return numeric != -1.0


def _sheet_target_uids(path: Path, *, train_folds_only: bool = False) -> set[str]:
    frame = pd.read_parquet(path)
    if train_folds_only and "original_fold" in frame.columns:
        frame = frame.loc[frame["original_fold"].map(_is_author_train_fold)].copy()
    if frame.empty:
        return set()
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
            protospacer = wt_value[4:24] if len(wt_value) >= 24 else wt_value
            uid = compute_target_uid(protospacer, wt_value)
            if uid:
                loci.add(uid)
        return loci
    return set()


def _collect_loci(specs: Iterable[_DatasetSpec] = _DATASET_SPECS) -> tuple[set[str], list[dict[str, str]]]:
    loci: set[str] = set()
    lineage: list[dict[str, str]] = []
    for spec in specs:
        dataset_dir = _STANDARDIZED / spec.study / spec.dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Standardized OptiPrime training dataset missing: {dataset_dir}"
            )
        dataset_loci: set[str] = set()
        for path in sorted(dataset_dir.glob("*.parquet")):
            dataset_loci |= _sheet_target_uids(path, train_folds_only=spec.train_folds_only)
        if not dataset_loci:
            raise ValueError(
                f"No training loci resolved for {spec.study}/{spec.dataset} "
                f"(OptiPrime vendor provenance)"
            )
        loci |= dataset_loci
        lineage.append({"study": spec.study, "dataset": spec.lineage_dataset})
    return loci, lineage


def sync_optiprime_vendor_provenance() -> dict[str, int]:
    """Populate manifest training metadata and loci sidecar for OptiPrime base."""
    loci, lineage = _collect_loci()
    if not loci:
        raise ValueError("No OptiPrime training loci resolved from standardized datasets")

    dataset_names = [entry["dataset"] for entry in lineage]
    training = {
        "dataset_source": "vendor",
        "dataset_name": " + ".join(dataset_names),
        "filters": {
            "study": sorted({entry["study"] for entry in lineage}),
            "dataset": dataset_names,
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
            "external_sources": [
                "PRIDICT library1 (Mathis et al. 2023, ref. 54)",
                "DeepPrime ClinVar train folds (Yu et al. 2023, ref. 55)",
                "PRIDICT2 library-diverse (Mathis et al. 2025, ref. 56)",
            ],
            "per_dataset_train_folds_only": {
                "deepprime-clinvar": True,
            },
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

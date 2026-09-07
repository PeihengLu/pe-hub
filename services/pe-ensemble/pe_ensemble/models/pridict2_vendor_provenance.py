"""Backfill training provenance for registered August PRIDICT2 vendor weights.

Paper Model A (``pridict1_1``): base-trained on **all** of PRIDICT library1,
then fine-tuned on library-diverse with fold ``run_x`` held out.

Paper Model B (``pridict1_2``): base-trained on all of library1 plus DeepPrime
ClinVar **train folds** (author ``original_fold == -1`` excluded), then the
same library-diverse fine-tune.

PRIDICT library1 has **no author test split** (``original_fold`` is unset on
every row). Vendor training therefore saw every library1 locus; leak checks
must record the full sheet, not a random holdout or the wrong library.

This module writes:

1. a ``training`` block on each vendor run ``manifest.json``
2. a ``train_target_loci.json`` sidecar for evaluation-time leak checks
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from pe_common.data_utils import TARGET_UID_COLUMN, compute_target_uid, target_uid_series

from . import weights_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STANDARDIZED = _REPO_ROOT / "datasets" / "standardized"

_RUN_RE = re.compile(r"__run_(\d+)(?:__|$)")
_VENDOR_PREFIXES = ("pridict1_1__", "pridict1_2__")


def parse_pridict2_vendor_run(weight_id: str) -> tuple[str, int]:
    """Return ``(lineage, held_out_fold)`` for an August vendor run id.

    Lineage is ``A`` (library1 base) or ``B`` (library1+ClinVar base).
    """
    if weight_id.startswith("pridict1_2__"):
        lineage = "B"
    elif weight_id.startswith("pridict1_1__"):
        lineage = "A"
    else:
        raise ValueError(f"Not a vendor PRIDICT2 Model A/B id: {weight_id!r}")
    match = _RUN_RE.search(weight_id)
    if match is None:
        raise ValueError(f"Cannot parse CV run from PRIDICT2 weight id {weight_id!r}")
    return lineage, int(match.group(1))


def _clean_uid(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"nan", "<NA>", "None"}:
        return None
    return text


def sheet_target_uids(
    path: Path,
    *,
    exclude_original_fold: Optional[float] = None,
    drop_author_test: bool = False,
) -> set[str]:
    """Collect target UIDs from one standardized parquet.

    Rows with missing ``original_fold`` are training data (library1 has no
    author split, so every row is kept). ``drop_author_test`` drops DeepPrime
    ``-1``. ``exclude_original_fold`` drops a PRIDICT2 CV held-out fold.
    """
    frame = pd.read_parquet(path)
    if "original_fold" in frame.columns and (
        exclude_original_fold is not None or drop_author_test
    ):
        fold = pd.to_numeric(frame["original_fold"], errors="coerce")
        keep = pd.Series(True, index=frame.index)
        if drop_author_test:
            keep &= ~fold.eq(-1.0)
        if exclude_original_fold is not None:
            keep &= ~fold.eq(float(exclude_original_fold))
        frame = frame.loc[keep].copy()
    if frame.empty:
        return set()
    if TARGET_UID_COLUMN in frame.columns:
        return {
            uid
            for uid in (_clean_uid(value) for value in frame[TARGET_UID_COLUMN].tolist())
            if uid
        }
    try:
        series = target_uid_series(frame)
        return {uid for uid in (_clean_uid(value) for value in series.tolist()) if uid}
    except Exception:
        pass
    if "wt_sequence" not in frame.columns:
        return set()
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


@lru_cache(maxsize=None)
def dataset_target_uids(
    study: str,
    dataset: str,
    *,
    exclude_original_fold: Optional[float] = None,
    drop_author_test: bool = False,
    standardized_root: Path = _STANDARDIZED,
) -> set[str]:
    dataset_dir = standardized_root / study / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Standardized dataset missing: {dataset_dir}")
    loci: set[str] = set()
    for path in sorted(dataset_dir.glob("*.parquet")):
        loci |= sheet_target_uids(
            path,
            exclude_original_fold=exclude_original_fold,
            drop_author_test=drop_author_test,
        )
    if not loci:
        raise ValueError(f"No training loci resolved for {study}/{dataset}")
    return loci


def collect_train_loci_for_run(weight_id: str, *, standardized_root: Path = _STANDARDIZED) -> tuple[set[str], list[dict[str, str]]]:
    lineage, held_out_fold = parse_pridict2_vendor_run(weight_id)
    loci: set[str] = set()
    lineage_entries: list[dict[str, str]] = []

    library1 = dataset_target_uids("pridict1", "library1", standardized_root=standardized_root)
    loci |= library1
    lineage_entries.append(
        {
            "study": "pridict1",
            "dataset": "library1",
            "note": "no author test split; all loci are training data",
        }
    )

    if lineage == "B":
        clinvar = dataset_target_uids(
            "deepprime",
            "deepprime_clinvar",
            drop_author_test=True,
            standardized_root=standardized_root,
        )
        loci |= clinvar
        lineage_entries.append(
            {
                "study": "deepprime",
                "dataset": "deepprime-clinvar",
                "note": "author original_fold=-1 excluded",
            }
        )

    diverse = dataset_target_uids(
        "pridict2",
        "library_diverse",
        exclude_original_fold=float(held_out_fold),
        standardized_root=standardized_root,
    )
    loci |= diverse
    lineage_entries.append(
        {
            "study": "pridict2",
            "dataset": "library-diverse",
            "note": f"testset_fold=={held_out_fold} held out",
        }
    )
    return loci, lineage_entries


def _training_block(
    *,
    lineage: str,
    held_out_fold: int,
    loci: set[str],
    lineage_entries: list[dict[str, str]],
) -> dict:
    dataset_names = [entry["dataset"] for entry in lineage_entries]
    return {
        "dataset_source": "vendor",
        "dataset_name": " + ".join(dataset_names),
        "filters": {
            "study": sorted({entry["study"] for entry in lineage_entries}),
            "dataset": dataset_names,
        },
        "split": {
            "use_original_fold": True,
            "original_fold_test_value": float(held_out_fold),
        },
        "hyperparameters": {},
        "model_kwargs": {},
        "data_provenance": {
            "target_uid_fingerprint": weights_registry.loci_fingerprint(loci),
            "n_target_loci": len(loci),
            "loci_recorded": True,
            "has_original_test_split": True,
            "library1_has_original_test_split": False,
            "library1_all_loci_are_train": True,
            "held_out_library_diverse_fold": held_out_fold,
            "includes_clinvar_train_folds": lineage == "B",
            "vendor_training_lineage": lineage_entries,
        },
    }


def _vendor_pridict2_ids() -> list[str]:
    ids: list[str] = []
    for entry in weights_registry.list_entries("pridict2"):
        if entry.get("source") != "vendor":
            continue
        weight_id = str(entry.get("id") or "")
        if weight_id.startswith(_VENDOR_PREFIXES):
            ids.append(weight_id)
    return sorted(ids)


def sync_pridict2_vendor_provenance() -> dict[str, int]:
    """Populate manifest training metadata and loci sidecars for August runs."""
    updated = 0
    n_loci_last = 0
    ids = _vendor_pridict2_ids()
    if not ids:
        raise FileNotFoundError("No vendor PRIDICT2 Model A/B weight sets registered")
    for weight_id in ids:
        lineage, held_out_fold = parse_pridict2_vendor_run(weight_id)
        loci, lineage_entries = collect_train_loci_for_run(weight_id)
        n_loci_last = len(loci)
        weights_registry.write_training_provenance(
            "pridict2",
            weight_id,
            training=_training_block(
                lineage=lineage,
                held_out_fold=held_out_fold,
                loci=loci,
                lineage_entries=lineage_entries,
            ),
            train_target_loci=sorted(loci),
        )
        updated += 1
    return {"updated": updated, "n_target_loci": n_loci_last}


if __name__ == "__main__":
    print(sync_pridict2_vendor_provenance())

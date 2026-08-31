"""Backfill training provenance for registered DeepPrime vendor weights.

The vendor checkpoints do not come with PE-Ensemble training metadata, but we
do know their source datasets:

- ``DeepPrime_base``: trained on DeepPrime ClinVar **train folds** (not ``Test``)
- ``DeepPrime_off``: trained on the DeepPrime off-target train folds
- ``DP_variant_*``: fine-tuned on one matching ``deepprime-small`` datasheet
  (train folds) after initializing from the ClinVar backbone

Author sheets label held-out rows with ``fold == "Test"`` (standardized as
``original_fold == -1``). Training loci sidecars must exclude those rows so
in-domain evaluation on the author test fold is not falsely flagged as leak.

This module converts that knowledge into:

1. a human-readable ``training`` block in each weight ``manifest.json``
2. a ``train_target_loci.json`` sidecar used by evaluation-time leak checks
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from pe_common.data_utils import compute_target_uid

from . import weights_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEEPPRIME_WORKBOOK = _REPO_ROOT / "datasets" / "raw" / "deepprime" / "deepprime-org.xlsx"

_PROTOSPACER_L = 4
_PROTOSPACER_R = 24

_DATASET_RENAMES = {
    "Library-ClinVar": "deepprime-clinvar",
    "Library-Small": "deepprime-small",
    "Library-Off": "deepprime-off",
    "Library-Off(sub-pool)": "deepprime-off-subpool",
}


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


@dataclass(frozen=True)
class _DeepPrimeSheetSpec:
    sheet_name: str
    dataset: str
    cell_line: str
    pe_system: str


_FINE_TUNE_DATASHEET_BY_WEIGHT: Dict[str, tuple[str, str]] = {
    "DP_variant_293T_NRCH_PE2_Opti_220428": ("hek293t", "nrch_pe2"),
    "DP_variant_293T_NRCH-PE2max_Opti_220815": ("hek293t", "nrch_pe2max"),
    "DP_variant_293T_PE2_Conv_220428": ("hek293t", "pe2"),
    "DP_variant_293T_PE2max_Opti_220428": ("hek293t", "pe2max"),
    "DP_variant_293T_PE2max_epegRNA_Opti_220428": ("hek293t", "pe2max_epegrna"),
    "DP_variant_293T_PE4max_Opti_220728": ("hek293t", "pe4max"),
    "DP_variant_293T_PE4max_epegRNA_Opti_220428": ("hek293t", "pe4max_epegrna"),
    "DP_variant_A549_PE2max_Opti_221114": ("a549", "pe2max"),
    "DP_variant_A549_PE2max_epegRNA_Opti_220428": ("a549", "pe2max_epegrna"),
    "DP_variant_A549_PE4max_Opti_220728": ("a549", "pe4max"),
    "DP_variant_A549_PE4max_epegRNA_Opti_220428": ("a549", "pe4max_epegrna"),
    "DP_variant_DLD1_NRCHPE4max_Opti_220728": ("dld1", "nrch_pe4max"),
    "DP_variant_DLD1_PE2max_Opti_221114": ("dld1", "pe2max"),
    "DP_variant_DLD1_PE4max_Opti_220728": ("dld1", "pe4max"),
    "DP_variant_HCT116_PE2_Opti_220428": ("hct116", "pe2"),
    "DP_variant_HeLa_PE2max_Opti_220815": ("hela", "pe2max"),
    "DP_variant_MDA_PE2_Opti_220428": ("mda_mb_231", "pe2"),
    "DP_variant_NIH_NRCHPE4max_Opti_220815": ("nih3t3", "nrch_pe4max"),
}


def _read_summary() -> list[_DeepPrimeSheetSpec]:
    summary = pd.read_excel(_DEEPPRIME_WORKBOOK, sheet_name="Summary", header=0)
    summary = summary.rename(columns={"Index": "Sheet name"})
    specs: list[_DeepPrimeSheetSpec] = []
    for _, row in summary.iterrows():
        dataset = _DATASET_RENAMES.get(str(row["Library"]).strip(), str(row["Library"]).strip())
        specs.append(
            _DeepPrimeSheetSpec(
                sheet_name=str(row["Sheet name"]),
                dataset=_normalize_name(dataset),
                cell_line=_normalize_name(str(row["Cell line"])),
                pe_system=_normalize_name(str(row["PE system"])),
            )
        )
    return specs


def _read_sheet(sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(_DEEPPRIME_WORKBOOK, sheet_name=sheet_name, skiprows=3, header=0)
    frame.columns = (
        frame.columns.str.replace(" ", "_").str.replace("\n", "").str.replace("\t", "")
    )
    frame.columns = frame.columns.str.lower()
    frame = frame.rename(
        columns={
            "wide_target_sequence(target_74bps_=_4bp_neighboring_sequence_+_20_bp_protospacer_+_3_bp_ngg_+_47_bp_neighboring_sequence)": "wt_sequence",
        }
    )
    return frame


def _is_author_train_fold(value: object) -> bool:
    """True for DeepPrime train folds (0–4); False for author ``Test`` / -1."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        # No fold label: treat as training (conservative for leak checks).
        return True
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"test", "-1"}:
            return False
        try:
            value = float(token)
        except ValueError:
            return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if pd.isna(numeric):
        return True
    return numeric != -1.0


def _frame_train_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop author held-out rows when a ``fold`` column is present."""
    if "fold" not in frame.columns or frame.empty:
        return frame
    mask = frame["fold"].map(_is_author_train_fold)
    return frame.loc[mask].reset_index(drop=True)


def _sheet_target_uids(frame: pd.DataFrame, *, train_only: bool = True) -> set[str]:
    working = _frame_train_rows(frame) if train_only else frame
    if "wt_sequence" not in working.columns:
        return set()
    wt_series = working["wt_sequence"].astype("string").fillna("")
    loci: set[str] = set()
    for wt in wt_series.tolist():
        wt_value = str(wt).strip().upper()
        if not wt_value:
            continue
        protospacer = wt_value[_PROTOSPACER_L:_PROTOSPACER_R]
        uid = compute_target_uid(protospacer, wt_value)
        if uid:
            loci.add(uid)
    return loci


def _sheet_has_author_test_split(frame: pd.DataFrame) -> bool:
    if "fold" not in frame.columns or frame.empty:
        return False
    return bool((~frame["fold"].map(_is_author_train_fold)).any())


def _collect_dataset_loci(
    specs: Iterable[_DeepPrimeSheetSpec],
) -> tuple[dict[tuple[str, str, str], set[str]], dict[tuple[str, str, str], bool]]:
    """Return train-fold loci and whether each sheet key had an author Test split."""
    loci_by_key: dict[tuple[str, str, str], set[str]] = {}
    has_test_by_key: dict[tuple[str, str, str], bool] = {}
    for spec in specs:
        key = (spec.dataset, spec.cell_line, spec.pe_system)
        frame = _read_sheet(spec.sheet_name)
        loci = _sheet_target_uids(frame, train_only=True)
        if not loci:
            continue
        loci_by_key[key] = loci
        has_test_by_key[key] = _sheet_has_author_test_split(frame)
    return loci_by_key, has_test_by_key


def _union_matching(
    loci_by_key: dict[tuple[str, str, str], set[str]],
    *,
    dataset: str,
    cell_line: Optional[str] = None,
    pe_system: Optional[str] = None,
) -> set[str]:
    merged: set[str] = set()
    for (entry_dataset, entry_cell_line, entry_pe_system), loci in loci_by_key.items():
        if entry_dataset != _normalize_name(dataset):
            continue
        if cell_line is not None and entry_cell_line != _normalize_name(cell_line):
            continue
        if pe_system is not None and entry_pe_system != _normalize_name(pe_system):
            continue
        merged.update(loci)
    return merged


def _any_has_test(
    has_test_by_key: dict[tuple[str, str, str], bool],
    *,
    dataset: str,
    cell_line: Optional[str] = None,
    pe_system: Optional[str] = None,
) -> bool:
    for (entry_dataset, entry_cell_line, entry_pe_system), has_test in has_test_by_key.items():
        if entry_dataset != _normalize_name(dataset):
            continue
        if cell_line is not None and entry_cell_line != _normalize_name(cell_line):
            continue
        if pe_system is not None and entry_pe_system != _normalize_name(pe_system):
            continue
        if has_test:
            return True
    return False


def _data_provenance(
    loci: set[str],
    *,
    lineage: list[dict[str, object]],
    has_original_test_split: bool,
) -> dict[str, object]:
    return {
        "target_uid_fingerprint": weights_registry.loci_fingerprint(loci),
        "n_target_loci": len(loci),
        "loci_recorded": bool(loci),
        "has_original_test_split": bool(has_original_test_split),
        "vendor_training_lineage": lineage,
        "train_folds_only": True,
    }


def _write_weight(
    weight_id: str,
    *,
    loci: set[str],
    filters: dict[str, object],
    dataset_name: str,
    lineage: list[dict[str, object]],
    has_original_test_split: bool,
) -> None:
    training = {
        "dataset_source": "vendor",
        "dataset_name": dataset_name,
        "filters": filters,
        "split": (
            {
                "use_original_fold": True,
                "original_fold_test_value": -1.0,
            }
            if has_original_test_split
            else None
        ),
        "hyperparameters": {},
        "model_kwargs": {},
        "data_provenance": _data_provenance(
            loci,
            lineage=lineage,
            has_original_test_split=has_original_test_split,
        ),
    }
    weights_registry.write_training_provenance(
        "deepprime",
        weight_id,
        training=training,
        train_target_loci=sorted(loci),
    )


def sync_deepprime_vendor_provenance() -> dict[str, int]:
    """Populate manifest training metadata and loci sidecars for DeepPrime."""
    if not _DEEPPRIME_WORKBOOK.is_file():
        raise FileNotFoundError(f"DeepPrime workbook not found: {_DEEPPRIME_WORKBOOK}")

    specs = _read_summary()
    loci_by_key, has_test_by_key = _collect_dataset_loci(specs)

    clinvar_loci = _union_matching(loci_by_key, dataset="deepprime-clinvar")
    off_loci = _union_matching(loci_by_key, dataset="deepprime-off")
    if not clinvar_loci:
        raise ValueError("No loci resolved for deepprime-clinvar")
    if not off_loci:
        raise ValueError("No loci resolved for deepprime-off")

    clinvar_has_test = _any_has_test(has_test_by_key, dataset="deepprime-clinvar")
    off_has_test = _any_has_test(has_test_by_key, dataset="deepprime-off")

    updated = 0

    _write_weight(
        "DeepPrime_base",
        loci=clinvar_loci,
        filters={"study": ["deepprime"], "dataset": ["deepprime-clinvar"]},
        dataset_name="deepprime-clinvar (train folds)",
        lineage=[{"study": "deepprime", "dataset": "deepprime-clinvar"}],
        has_original_test_split=clinvar_has_test,
    )
    updated += 1

    _write_weight(
        "DeepPrime_off",
        loci=off_loci,
        filters={"study": ["deepprime"], "dataset": ["deepprime-off"]},
        dataset_name="deepprime-off (train folds)",
        lineage=[{"study": "deepprime", "dataset": "deepprime-off"}],
        has_original_test_split=off_has_test,
    )
    updated += 1

    for weight_id, (cell_line, pe_system) in _FINE_TUNE_DATASHEET_BY_WEIGHT.items():
        small_loci = _union_matching(
            loci_by_key,
            dataset="deepprime-small",
            cell_line=cell_line,
            pe_system=pe_system,
        )
        if not small_loci:
            raise ValueError(
                f"No deepprime-small loci resolved for {weight_id} "
                f"({cell_line}-{pe_system})"
            )
        small_has_test = _any_has_test(
            has_test_by_key,
            dataset="deepprime-small",
            cell_line=cell_line,
            pe_system=pe_system,
        )
        lineage = [
            {"study": "deepprime", "dataset": "deepprime-clinvar"},
            {
                "study": "deepprime",
                "dataset": "deepprime-small",
                "cell_line": cell_line,
                "pe_system": pe_system,
            },
        ]
        _write_weight(
            weight_id,
            loci=clinvar_loci | small_loci,
            filters={
                "study": ["deepprime"],
                "dataset": ["deepprime-clinvar", "deepprime-small"],
                "cell_line": [cell_line],
                "pe_system": [pe_system],
            },
            dataset_name=(
                f"deepprime-clinvar + deepprime-small/{cell_line}-{pe_system} "
                "(train folds)"
            ),
            lineage=lineage,
            has_original_test_split=clinvar_has_test or small_has_test,
        )
        updated += 1

    return {"updated_weights": updated}


if __name__ == "__main__":
    print(sync_deepprime_vendor_provenance())

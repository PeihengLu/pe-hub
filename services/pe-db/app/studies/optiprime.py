from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import (
    align_wt_mut_sequences,
    reverse_complement,
    shift_coords_after_indel_pad,
)

from ..catalog.records import DatasheetScaffoldAssignment
from ..catalog.scaffolds import (
    MINSEPIE_DATASET_SCAFFOLD_ID,
    SCAFFOLD_ID_CONVENTIONAL,
    SCAFFOLD_ID_OPTIPRIME_BLPI_FE,
    default_scaffold_for_pridict,
    scaffold_id_from_deepprime_label,
)
from ..catalog.studies import get_dataset_record
from ..config import get_settings
from ..pipeline.names import _normalize_name
from ..pipeline.registry import StudyPipeline, register_study
from ..pipeline.schema import (
    _attach_endo_coordinate_columns,
    _build_standardized_output_df,
    _coerce_original_fold,
    _shift_coords_after_wt_mut_align,
    _write_partial_standardized_output,
    endo_standard_columns,
)
from ..utils.deepspcas9 import fill_missing_spcas9_scores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OptiPrime (Hsu et al. 2026)
# ---------------------------------------------------------------------------

_OPTIPRIME_XLSX = "41587_2026_3261_MOESM3_ESM.xlsx"
_OPTIPRIME_CONDITIONS = [
    ("hek293t", "pe2", "HEK293T_PE2_editing"),
    ("hek293t", "pe4", "HEK293T_PE4_editing"),
    ("hela", "pe2", "HeLa_PE2_editing"),
    ("hela", "pe4", "HeLa_PE4_editing"),
]
# Lib-MMR Design category == "Endogenous" are positive-control pegRNAs previously
# used at endogenous loci, but assayed here in the same lentiviral reporter screen.
_OPTIPRIME_LIBMMR_ENDO_CATEGORY = "Endogenous"


def _optiprime_build_export_frame(
    sub: pd.DataFrame,
    *,
    wt_col: str,
    mut_col: str,
    spacer_col: str,
    pbs_col: str,
    hom_col: str,
    eff_col: str,
) -> pd.DataFrame:
    """Map OptiPrime supplementary columns into the shared export CSV schema."""
    exported = pd.DataFrame()
    exported["wt_sequence"] = sub[wt_col].astype(str).str.upper().str.strip()
    exported["mut_sequence"] = sub[mut_col].astype(str).str.upper().str.strip()
    exported["spacer"] = sub[spacer_col].astype(str).str.strip()
    exported["pbs"] = sub[pbs_col].astype(str).str.strip()
    exported["homology_arm"] = sub[hom_col].astype(str).str.strip()
    exported["measured_pe_efficiency"] = pd.to_numeric(sub[eff_col], errors="coerce")
    exported["deepspcas9_score"] = 0.0

    wt_seqs = exported["wt_sequence"]
    mut_seqs = exported["mut_sequence"]
    wt_lens = wt_seqs.str.len()
    mut_lens = mut_seqs.str.len()
    exported["type_sub"] = wt_lens == mut_lens
    exported["type_ins"] = wt_lens < mut_lens
    exported["type_del"] = wt_lens > mut_lens
    exported["edit_len"] = (mut_lens - wt_lens).abs().clip(lower=1)

    sub_mask = exported["type_sub"]
    if sub_mask.any():
        def _count_diffs(row: pd.Series) -> int:
            return sum(1 for a, b in zip(row["wt_sequence"], row["mut_sequence"]) if a != b)

        exported.loc[sub_mask, "edit_len"] = exported.loc[sub_mask].apply(_count_diffs, axis=1)
    return exported


def _optiprime_write_condition_csvs(
    df: pd.DataFrame,
    *,
    dataset_key: str,
    wt_col: str,
    mut_col: str,
    spacer_col: str,
    pbs_col: str,
    hom_col: str,
) -> None:
    """Write one CSV per cell-line / PE-system condition for an OptiPrime dataset."""
    for cell_line, pe_system, eff_col in _OPTIPRIME_CONDITIONS:
        out_dir = DATA_ROOT / "exported" / "optiprime" / dataset_key.replace("_", "-")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cell_line}-{pe_system}.csv"

        mask = pd.to_numeric(df[eff_col], errors="coerce").notna()
        sub = df.loc[mask].copy()
        if sub.empty:
            logger.warning("No valid rows for %s %s-%s", dataset_key, cell_line, pe_system)
            continue

        exported = _optiprime_build_export_frame(
            sub,
            wt_col=wt_col,
            mut_col=mut_col,
            spacer_col=spacer_col,
            pbs_col=pbs_col,
            hom_col=hom_col,
            eff_col=eff_col,
        )
        exported.to_csv(out_path, index=False)
        logger.info("Exported %s rows to %s", len(exported), out_path)


def _export_optiprime_datasheets() -> None:
    """Export OptiPrime Lib-MMR / Lib-CV tables, splitting endogenous controls out of Lib-MMR."""
    xlsx_path = DATA_ROOT / "raw" / "optiprime" / _OPTIPRIME_XLSX
    if not xlsx_path.exists():
        logger.warning("OptiPrime supplementary XLSX not found at %s", xlsx_path)
        return

    # --- Lib-MMR: split Design category == Endogenous into lib-mmr-controls ---
    lib_mmr = pd.read_excel(xlsx_path, sheet_name="Supp Table 4 LibMMR", header=0)
    logger.info("Read %s rows from Supp Table 4 LibMMR", len(lib_mmr))
    category = lib_mmr["Design category"].astype(str).str.strip()
    endo_mask = category.eq(_OPTIPRIME_LIBMMR_ENDO_CATEGORY)
    lib_mmr_cols = dict(
        wt_col="Designed target (ps-pam-edit)",
        mut_col="Designed edited target (ps-pam-edit)",
        spacer_col="Designed 5G pegRNA spacer",
        pbs_col="PBS",
        hom_col="Homology arm",
    )
    _optiprime_write_condition_csvs(
        lib_mmr.loc[~endo_mask].copy(),
        dataset_key="lib_mmr",
        **lib_mmr_cols,
    )
    _optiprime_write_condition_csvs(
        lib_mmr.loc[endo_mask].copy(),
        dataset_key="lib_mmr_controls",
        **lib_mmr_cols,
    )
    logger.info(
        "Lib-MMR split: %s primary rows, %s endogenous-control rows",
        int((~endo_mask).sum()),
        int(endo_mask.sum()),
    )

    # --- Lib-CV ---
    lib_cv = pd.read_excel(xlsx_path, sheet_name="Supp Table 5 LibCV", header=0)
    logger.info("Read %s rows from Supp Table 5 LibCV", len(lib_cv))
    _optiprime_write_condition_csvs(
        lib_cv,
        dataset_key="lib_cv",
        wt_col="unedited_target",
        mut_col="edited_target",
        spacer_col="spacer",
        pbs_col="pbs_bind",
        hom_col="homology_arm",
    )


def _optiprime_genomic_protospacer_probes(spacer: str) -> list[str]:
    """20-mer probes for Hsu's designed 5G pegRNA spacer column.

    The DNA protospacer is 20 nt. A 21-nt spacer that starts with G is the
    U6 5′ G plus that 20-mer (``spacer[1:]``), not the first 20 of the pegRNA.
    """
    spacer = str(spacer).upper().replace("U", "T")
    probes: list[str] = []
    if len(spacer) == 21 and spacer.startswith("G"):
        probes.append(spacer[1:])
    if len(spacer) >= 20:
        probes.append(spacer[-20:])
        if spacer[:20] not in probes:
            probes.append(spacer[:20])
    elif spacer:
        probes.append(spacer)
    return [probe for probe in probes if len(probe) >= 15]


def _locate_optiprime_protospacer(wt_sequence: str, spacer: str) -> tuple[int, int]:
    """Return 20 bp protospacer bounds on an OptiPrime ``ps-pam-edit`` (or lib-cv) WT.

    Author targets often start at the spacer (Lib-MMR has no DeepPrime 4 bp pad).
    Designed 5G spacers may carry an extra leading G; match the genomic 20-mer.
    """
    wt = str(wt_sequence).upper().replace("U", "T")
    for window in _optiprime_genomic_protospacer_probes(spacer):
        idx = wt.find(window)
        if idx >= 0:
            return idx, idx + len(window)
    return 0, min(20, len(wt))


def _optiprime_homology_end(
    mut_sequence: str,
    *,
    nick: int,
    edit_pos: int,
    edit_len: int,
    type_del: bool,
    homology_arm: str,
) -> int:
    """Right bound of the pegRNA RTT / 3′ homology on the unaligned Mut sequence."""
    mut = str(mut_sequence).upper().replace("U", "T")
    if homology_arm is None or (isinstance(homology_arm, float) and pd.isna(homology_arm)):
        ha = ""
    else:
        ha = str(homology_arm).upper().replace("U", "T").strip()
    if ha in {"", "NAN", "NONE", "<NA>"}:
        ha = ""
    ha_start = int(edit_pos) if type_del else int(edit_pos) + int(edit_len)
    ha_start = max(int(nick), ha_start)
    if ha:
        idx = mut.find(ha, ha_start)
        if idx < 0:
            idx = mut.find(ha, int(nick))
        if idx >= 0:
            return idx + len(ha)
        return min(len(mut), ha_start + len(ha))
    return len(mut)


def _standardize_optiprime(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Standardize OptiPrime Lib-MMR / Lib-MMR-controls / Lib-CV data to the shared PE schema."""
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    output_name = f"{cell_line}-{pe_system}.parquet"

    logger.info(
        "Standardizing optiprime dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset, cell_line, pe_system, len(data),
    )

    df = data.copy()

    # Derive edit type flags
    mutation_flags = df[["type_sub", "type_ins", "type_del"]].fillna(False).astype(bool)
    valid_mask = mutation_flags.any(axis=1)
    df = df.loc[valid_mask].reset_index(drop=True)
    mutation_flags = df[["type_sub", "type_ins", "type_del"]].fillna(False).astype(bool)
    df[["type_sub", "type_ins", "type_del"]] = mutation_flags
    df["mut_type"] = np.select(
        [mutation_flags["type_sub"], mutation_flags["type_ins"], mutation_flags["type_del"]],
        [0, 1, 2], default=-1,
    )

    # ps-pam-edit starts at the 20 bp spacer (no DeepPrime 4 bp pad).
    wt_sequence = df["wt_sequence"].astype(str).str.upper()
    mut_sequence = df["mut_sequence"].astype(str).str.upper()
    spacer_col = df["spacer"].astype(str) if "spacer" in df.columns else pd.Series("", index=df.index)
    prot_bounds = [
        _locate_optiprime_protospacer(wt, spacer)
        for wt, spacer in zip(wt_sequence, spacer_col)
    ]
    protospacer_l = pd.Series([left for left, _ in prot_bounds], index=df.index, dtype=int)
    protospacer_r = pd.Series([right for _, right in prot_bounds], index=df.index, dtype=int)
    protospacer = pd.Series(
        [wt[int(left):int(right)] for wt, left, right in zip(wt_sequence, protospacer_l, protospacer_r)],
        index=df.index,
    )
    df["group_id"] = protospacer.map(hash).groupby(protospacer).ngroup()

    pbs_len = df["pbs"].astype(str).str.upper().str.len()
    nick_pos = protospacer_r - 3
    df["pbs_l"] = nick_pos - pbs_len
    df["pbs_r"] = nick_pos

    df["rtt_l"] = nick_pos

    def _find_edit_pos(wt, mut):
        for i, (a, b) in enumerate(zip(wt, mut)):
            if a != b:
                return i
        return min(len(wt), len(mut))

    edit_positions = pd.Series(
        [_find_edit_pos(w, m) for w, m in zip(wt_sequence, mut_sequence)],
        index=df.index,
        dtype=int,
    )
    df["lha_l"] = nick_pos
    df["lha_r"] = edit_positions

    edit_len = df["edit_len"].astype(int)
    if "homology_arm" in df.columns:
        homology_arm = df["homology_arm"].where(df["homology_arm"].notna(), "").astype(str)
    else:
        homology_arm = pd.Series("", index=df.index)
    rtt_r = [
        _optiprime_homology_end(
            mut,
            nick=int(nick),
            edit_pos=int(edit_pos),
            edit_len=int(length),
            type_del=bool(is_del),
            homology_arm=ha,
        )
        for mut, nick, edit_pos, length, is_del, ha in zip(
            mut_sequence,
            nick_pos,
            edit_positions,
            edit_len,
            df["type_del"],
            homology_arm,
        )
    ]
    df["rtt_r"] = pd.Series(rtt_r, index=df.index, dtype=int)
    df["rha_l"] = edit_positions + np.where(df["type_del"], 0, edit_len)
    df["rha_r"] = df["rtt_r"]

    def _align_row(row):
        return align_wt_mut_sequences(
            str(row["wt_sequence"]),
            str(row["mut_sequence"]),
            int(row["lha_r"]),
            edit_length=int(row["edit_len"]),
            edit_type=int(row["mut_type"]),
        )

    aligned = df.apply(_align_row, axis=1, result_type="expand")
    aligned.columns = ["wt_aligned", "mut_aligned"]
    coords = _shift_coords_after_wt_mut_align(
        index=df.index,
        type_ins=df["type_ins"],
        type_del=df["type_del"],
        edit_len=edit_len,
        lha_r=df["lha_r"],
        protospacer_l=protospacer_l,
        protospacer_r=protospacer_r,
        pbs_l=df["pbs_l"],
        pbs_r=df["pbs_r"],
        rtt_l=df["rtt_l"],
        rtt_r=df["rtt_r"],
        lha_l=df["lha_l"],
        rha_l=df["rha_l"],
        rha_r=df["rha_r"],
    )

    output_df = _build_standardized_output_df(
        df["group_id"], df["type_sub"], df["type_ins"], df["type_del"], df["edit_len"],
        aligned["wt_aligned"], aligned["mut_aligned"],
        coords["protospacer_l"], coords["protospacer_r"],
        coords["pbs_l"], coords["pbs_r"], coords["rtt_l"], coords["rtt_r"],
        coords["lha_l"], coords["lha_r"], coords["rha_l"], coords["rha_r"],
        df["deepspcas9_score"], df["measured_pe_efficiency"],
        original_fold=None,
    )

    output_path = DATA_ROOT / "standardized" / "optiprime" / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized optiprime data: %s (%s rows)", output_path, len(output_df))


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_optiprime_scaffold_assignments
    return build_optiprime_scaffold_assignments()


register_study(StudyPipeline(
    key="optiprime",
    exporters=(_export_optiprime_datasheets,),
    standardizers={
        "lib_mmr": _standardize_optiprime,
        "lib_mmr_controls": _standardize_optiprime,
        "lib_cv": _standardize_optiprime,
    },
    scaffold_assignments=_scaffold_assignments,
))

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

def _export_deepprime_datasheets() -> None:
    """
    Export all sheets from the original DeepPrime Excel file to CSV format.
    
    Args:
        original_excel_path: Path to the original DeepPrime Excel file.
                            If None, uses default path in raw/deepprime-org/
    """
    original_excel_path = DATA_ROOT / "raw" / "deepprime" / "deepprime-org.xlsx"
    # Summary sheet is the catalog for the individual experiment sheets.
    original_data_catalog = pd.read_excel(
        original_excel_path, sheet_name="Summary", header=0,
    )
    original_data_catalog.rename(columns={"Index": "Sheet name"}, inplace=True)

    logger.info("Processing %s DeepPrime sheets", len(original_data_catalog))

    dataset_renames = {
        "Library-ClinVar": "deepprime-clinvar",
        "Library-Small": "deepprime-small",
        "Library-Off": "deepprime-off",
        "Library-Off(sub-pool)": "deepprime-off-subpool",
    }

    def _read_from_deepprime_org(excel_path: Path, sheet_name: str = "1") -> pd.DataFrame:
        """
        Read from the DeepPrime original excel file
        
        Args:
            excel_path: Path to the original deep prime Excel file
            sheet_name: Sheet name to read from
            
        Returns:
            DataFrame with cleaned column names
        """
        # Skip metadata rows and use the 4th row as the header.
        original_data = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=3, header=0)

        # Normalize column names for downstream usage.
        original_data.columns = (original_data.columns
                                .str.replace(" ", "_")
                                .str.replace("\n", "")
                                .str.replace("\t", ""))

        original_data.columns = original_data.columns.str.lower()

        # Rename long opaque columns used in sequence processing.
        original_data.rename(columns={
            "wide_target_sequence(target_74bps_=_4bp_neighboring_sequence_+_20_bp_protospacer_+_3_bp_ngg_+_47_bp_neighboring_sequence)": "wt_sequence",
            "edited_target_sequence(target_74bps_=_rt-pbs_corresponding_region_and_masked_by_'x')": "mut_sequence",
        }, inplace=True)

        return original_data

    for _, row in original_data_catalog.iterrows():
        sheet_name = str(row["Sheet name"])
        cell_line = _normalize_name(str(row["Cell line"]))
        pe_system = _normalize_name(str(row["PE system"]))
        dataset = str(row["Library"])
        if dataset in dataset_renames:
            dataset = dataset_renames[dataset]

        logger.debug(
            "DeepPrime sheet=%s dataset=%s cell_line=%s pe_system=%s",
            sheet_name,
            dataset,
            cell_line,
            pe_system,
        )

        original_data = _read_from_deepprime_org(
            original_excel_path, sheet_name=sheet_name
        )

        output_path = (
            DATA_ROOT / "exported" / "deepprime" / dataset / f"{cell_line}-{pe_system}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_data.to_csv(output_path, index=False)
        logger.info("Saved DeepPrime datasheet: %s", output_path)


def _standardize_deepprime_ontarget(
    data: Optional[pd.DataFrame],
    cell_line: str,
    pe_system: str,
    dataset: str,
    *,
    study_key: str = "deepprime",
) -> None:
    """
    Standardize DeepPrime-style on-target datasets to the shared PE schema.

    Also used for DeepPE after ``_prepare_deeppe_export_df`` maps Kim et al. exports
    into the same masked-sequence column layout.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    study_key = _normalize_name(study_key)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'deepprime' / dataset / input_name)

    logger.info(
        "Standardizing %s dataset=%s cell_line=%s pe_system=%s rows=%s",
        study_key,
        dataset,
        cell_line,
        pe_system,
        len(data),
    )
    df = data.copy()

    # ---- Step 1: Determine mutation type and filter invalid rows ----
    # Keep the one-hot booleans for output; derive an integer mut_type for internal calculations
    mutation_flags = df[['type_sub', 'type_ins', 'type_del']].fillna(False).astype(bool)
    valid_mutation_mask = pd.Series(mutation_flags.any(axis=1), index=df.index, dtype=bool)
    df = df.loc[valid_mutation_mask].reset_index(drop=True)
    mutation_flags = df[['type_sub', 'type_ins', 'type_del']].fillna(False).astype(bool)
    df[['type_sub', 'type_ins', 'type_del']] = mutation_flags
    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    df['mut_type'] = np.select(
        [mutation_flags['type_sub'], mutation_flags['type_ins'], mutation_flags['type_del']],
        [0, 1, 2],
        default=-1
    )

    # ---- Step 2: Compute protospacer and assign group IDs ----
    # Group rows with identical protospacers to prevent data leakage
    # deepprime sequence starts from 4bp upstream of the 20bp protospacer
    PROTOSPACER_L, PROTOSPACER_R = 4, 24  # 0-indexed
    wt_sequence = pd.Series(df['wt_sequence'], dtype='string')
    protospacer = wt_sequence.map(
        lambda seq: seq[PROTOSPACER_L:PROTOSPACER_R] if isinstance(seq, str) else ''
    )
    df['protospacer'] = protospacer
    # group rows by protospacer and save the group id
    df['group_id'] = df.groupby('protospacer').ngroup()

    # ---- Step 3: Compute PBS and LHA locations ----
    # In DeepPrime's format, mut_sequence uses leading 'x'/'X' characters to mask
    # positions upstream of the PBS that are not involved in the editing process.
    # The PBS left boundary is therefore the index of the first non-mask character.
    mut_sequence = pd.Series(df['mut_sequence'], dtype='string').fillna('')
    df['pbs_l'] = mut_sequence.map(lambda seq: len(seq) - len(seq.lstrip('xX')))
    df['pbs_r'] = df['pbs_l'] + df['pbslen']
    df['lha_l'] = df['pbs_r']
    df['lha_r'] = np.where(
        df['type_del'],
        df['pbs_r'] + (df['rt-pbslen'] - df['pbslen'] - df['rha_len']),
        df['pbs_r'] + (df['rt-pbslen'] - df['pbslen'] - df['rha_len'] - df['edit_len'])
    )

    # ---- Step 4: Compute RHA and RTT locations ----
    # The wt offset is +edit_len for deletion, -edit_len for insertion, 0 for substitution
    base = df['pbs_l'] + df['rt-pbslen']
    wt_offset = np.select(
        [df['type_del'], df['type_ins']],
        [df['edit_len'], -df['edit_len']],
        default=0
    )
    rha_wt_r = base + wt_offset

    df['rha_l'] = base - df['rha_len'] + wt_offset
    df['rha_r'] = np.where(df['type_del'], rha_wt_r, base)
    df['rtt_l'] = df['pbs_r']
    df['rtt_r'] = np.where(df['type_del'], rha_wt_r, base)

    # ---- Step 5: Reconstruct mutated sequences and align with wt ----
    # Remove the masking from the mutated sequence
    df['_rha_wt_r'] = rha_wt_r  # store intermediate for per-row string ops

    def _reconstruct_and_align(row):
        """
        Reconstruct the unmasked mutated sequence from the wild type
        and align with the wild type sequence
        """
        wt = str(row['wt_sequence'])
        lha_r = int(row['lha_r'])
        rha_wt_r_val = int(row['_rha_wt_r'])
        edit_len = int(row['edit_len'])
        type_sub = bool(row['type_sub'])
        type_ins = bool(row['type_ins'])
        type_del = bool(row['type_del'])

        # Reconstruct the unmasked mutated sequence by injecting the observed
        # RT-PBS segment from mut_sequence into WT context.
        masked_mut = str(row['mut_sequence'])
        pbs_l = int(row['pbs_l'])
        rt_pbs_len = int(row['rt-pbslen'])
        rt_pbs_right = pbs_l + rt_pbs_len
        observed_rt_pbs = masked_mut[pbs_l:rt_pbs_right].upper().replace("U", "T")

        if type_ins:
            wt_window_len = max(rt_pbs_len - edit_len, 0)
        elif type_del:
            wt_window_len = rt_pbs_len + edit_len
        else:
            wt_window_len = rt_pbs_len
        wt_suffix_start = min(max(pbs_l + wt_window_len, 0), len(wt))
        mut = wt[:pbs_l] + observed_rt_pbs + wt[wt_suffix_start:]

        # Pad with N at the edit position to align wt/mut to the same length
        mut_type = row['mut_type']
        return align_wt_mut_sequences(
                wt, mut, lha_r, edit_length=edit_len, edit_type=mut_type)

    aligned = df.apply(_reconstruct_and_align, axis=1, result_type='expand')
    aligned.columns = ['wt_aligned', 'mut_aligned']

    coords = _shift_coords_after_wt_mut_align(
        index=df.index,
        type_ins=df['type_ins'],
        type_del=df['type_del'],
        edit_len=df['edit_len'],
        lha_r=df['lha_r'],
        protospacer_l=PROTOSPACER_L,
        protospacer_r=PROTOSPACER_R,
        pbs_l=df['pbs_l'],
        pbs_r=df['pbs_r'],
        rtt_l=df['rtt_l'],
        rtt_r=df['rtt_r'],
        lha_l=df['lha_l'],
        rha_l=df['rha_l'],
        rha_r=df['rha_r'],
    )

    # ---- Step 6: Build output DataFrame ----
    if "fold" in df.columns:
        original_fold = pd.to_numeric(
            df["fold"].replace("Test", -1),
            errors="coerce",
        )
    else:
        original_fold = None
    output_df = _build_standardized_output_df(
        df['group_id'], df['type_sub'], df['type_ins'], df['type_del'], df['edit_len'],
        aligned['wt_aligned'], aligned['mut_aligned'],
        coords['protospacer_l'], coords['protospacer_r'],
        coords['pbs_l'], coords['pbs_r'], coords['rtt_l'], coords['rtt_r'],
        coords['lha_l'], coords['lha_r'], coords['rha_l'], coords['rha_r'],
        df['deepspcas9_score'], df['measured_pe_efficiency'], original_fold)

    # export the data to a parquet file
    output_path = DATA_ROOT / "standardized" / study_key / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized %s data: %s", study_key, output_path)


def _standardize_deepprime_off(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
    *,
    study_key: str = "deepprime",
) -> None:
    """Partial standardization for DeepPrime Off-subpool mismatch tables.

    Sequence-complete Library-Off PE2 data is handled by
    ``_standardize_deepprime_ontarget``. Subpool sheets typically expose
    ``edit_type`` (``transv``, ``transi``, ``transv2``, ``Ins``, ``Del``) or
    type flags without recoverable 74 bp target sequences.
    """
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if {"type_sub", "type_ins", "type_del"}.issubset(data.columns):
        type_sub = data["type_sub"].astype(bool)
        type_ins = data["type_ins"].astype(bool)
        type_del = data["type_del"].astype(bool)
    elif "edit_type" in data.columns:
        labels = data["edit_type"].astype("string").str.strip().str.lower()
        # Mismatch-screen categories; mapped to PE-DB sub/ins/del for statistics only.
        type_sub = labels.isin({"transv", "transi", "transv2"})
        type_ins = labels.isin({"ins", "insertion"})
        type_del = labels.isin({"del", "deletion"})
    else:
        raise ValueError(
            f"DeepPrime off-target export for {dataset} {cell_line}-{pe_system} "
            "must include type_sub/type_ins/type_del or edit_type."
        )

    if "edit_length" in data.columns:
        edit_len = pd.to_numeric(data["edit_length"], errors="coerce")
    elif "edit_len" in data.columns:
        edit_len = pd.to_numeric(data["edit_len"], errors="coerce")
    else:
        edit_len = pd.Series(np.nan, index=data.index, dtype=float)

    if "avg.on-target_efficiency" in data.columns:
        efficiency_col = "avg.on-target_efficiency"
    else:
        efficiency_col = None
        for candidate in ("editing_efficiency", "measured_pe_efficiency"):
            if candidate in data.columns:
                efficiency_col = candidate
                break
    if efficiency_col is None:
        editing_efficiency = pd.Series(0.0, index=data.index, dtype=float)
    else:
        editing_efficiency = pd.to_numeric(data[efficiency_col], errors="coerce").fillna(0.0)

    _write_partial_standardized_output(
        pd.DataFrame(
            {
                "type_sub": type_sub,
                "type_ins": type_ins,
                "type_del": type_del,
                "edit_len": edit_len,
                "editing_efficiency": editing_efficiency.astype(float),
            }
        ),
        study=study_key,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )


def _standardize_deepprime_datasheet(data, cell_line, pe_system, dataset):
    if dataset == "deepprime_off":
        if {"wt_sequence", "mut_sequence"}.issubset(data.columns):
            _standardize_deepprime_ontarget(data, cell_line, pe_system, dataset)
            return
        raise ValueError(
            f"DeepPrime-Off datasheet {cell_line}-{pe_system} lacks "
            "wt_sequence/mut_sequence and cannot be fully standardized."
        )
    if dataset == "deepprime_off_subpool":
        _standardize_deepprime_off(data, cell_line, pe_system, dataset)
        return
    _standardize_deepprime_ontarget(data, cell_line, pe_system, dataset)


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_deepprime_scaffold_assignments
    return build_deepprime_scaffold_assignments(data_root)


register_study(StudyPipeline(
    key="deepprime",
    exporters=(_export_deepprime_datasheets,),
    standardizers={
        "deepprime_clinvar": _standardize_deepprime_datasheet,
        "deepprime_small": _standardize_deepprime_datasheet,
        "deepprime_off": _standardize_deepprime_datasheet,
        "deepprime_off_subpool": _standardize_deepprime_datasheet,
    },
    scaffold_assignments=_scaffold_assignments,
))

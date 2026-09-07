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

from ._pridict import (
    _attach_pridict_outcome_distribution,
    _parse_pridict_location_column,
)

def _export_pridict2_library_diverse_datasheets() -> None:
    """
    Export the PRIDICT2 Library Diverse datasheet (supplementary table 2).
    """
    original_excel_path = DATA_ROOT / "raw" / "pridict2" / "pridict2-org.xlsx"
    # Sheet 2 contains library diverse
    library_diverse_data = pd.read_excel(original_excel_path, sheet_name=2, header=0)
    efficiency_columns = [col for col in library_diverse_data.columns if 'averageedited' in col]
    # library diverse contains editing data for four cell lines: HEK, K562, K562MLH1dn, AdV
    # each cell line has a separate sheet
    for cell_line in ['HEK', 'K562', 'K562MLH1dn', 'AdV']:
        efficiency_column = f'{cell_line}averageedited'
        data = library_diverse_data[efficiency_column]
        # concatenate all columns not containing any efficiency column
        information_columns = [col for col in library_diverse_data.columns if col not in efficiency_columns]
        data = pd.concat([data] + [library_diverse_data[col] for col in information_columns], axis=1)

        # remove columns where editing efficiency is NaN
        data = data.dropna(subset=[efficiency_column])

        cell_line_norm = cell_line.lower().replace("-", "_")
        if cell_line == 'AdV':
            dataset_name = 'library-diverse-invivo'
            output_path = (
                DATA_ROOT / 'exported' / 'pridict2' / dataset_name /
                f'{cell_line_norm}-pe2.csv'
            )
        else:
            dataset_name = 'library-diverse'
            output_path = (
                DATA_ROOT / 'exported' / 'pridict2' / dataset_name /
                f'{cell_line_norm}-pe2.csv'
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Saved PRIDICT2 library diverse data for {cell_line} to {output_path}")


def _export_pridict2_endogenous_datasheets() -> None:
    """
    Export PRIDICT2 TRIP endogenous editing survey (Mathis et al., Nat. Biotechnol. 2024).

    Source: ``pridict2-org.xlsx`` supplementary table 12 — TRIP library editing
    results at endogenous genomic sites (Fig. 3, Ext. Fig. 4–5). Per the publication
    and ePRIDICT training data, TRIP was performed in K562 with prime editor PE2.
    """
    original_excel_path = DATA_ROOT / "raw" / "pridict2" / "pridict2-org.xlsx"
    trip_df = pd.read_excel(original_excel_path, sheet_name="12", header=0)

    cell_line = "k562"
    pe_system = "pe2"
    out_dir = DATA_ROOT / "exported" / "pridict2" / "trip_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{cell_line}-{pe_system}.csv"
    trip_df.to_csv(output_path, index=False)
    logger.info(
        "Saved PRIDICT2 TRIP analysis (%s-%s): %s (%s rows)",
        cell_line,
        pe_system,
        output_path,
        len(trip_df),
    )


def _standardize_pridict2_library_diverse(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT2 library-diverse data to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'pridict2' / dataset / input_name)
    logger.info(
        "Standardizing PRIDICT2 dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset,
        cell_line,
        pe_system,
        len(data),
    )

    # ---- Step 1: calculate group ID based on the spacer column
    df = data.copy()
    df['group_id'] = df.groupby('spacer').ngroup()

    # ---- Step 2: determine mutation type and edit length ----
    correction_type = pd.Series(df['Correction_Type'], copy=False).astype('string').str.strip().str.lower()
    type_sub = correction_type.eq('replacement')
    type_ins = correction_type.eq('insertion')
    type_del = correction_type.eq('deletion')

    unknown_mask = ~(type_sub | type_ins | type_del)
    if unknown_mask.any():
        unknown_values = df.loc[unknown_mask, 'Correction_Type'].astype(str).unique().tolist()[:5]
        print(unknown_values)
        raise ValueError(f"Unsupported Correction_Type values: {unknown_values}")

    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    edit_type = pd.Series(
        np.select([type_sub, type_ins, type_del], [0, 1, 2], default=-1),
        index=df.index,
    ).astype(int)
    edit_len = pd.Series(pd.to_numeric(df['Correction_Length'], errors='raise'), index=df.index).astype(int)

    # ---- Step 3: compute protospacer / PBS / RTT / LHA / RHA locations ----
    wt_sequence = pd.Series(df['wide_initial_target'], copy=False).astype('string').str.upper()
    mut_sequence = pd.Series(df['wide_mutated_target'], copy=False).astype('string').str.upper()

    protospacer_l, protospacer_r = _parse_pridict_location_column(
        pd.Series(df['protospacerlocation_only_initial'], copy=False), 'protospacerlocation_only_initial'
    )
    pbs_l, pbs_r = _parse_pridict_location_column(pd.Series(df['PBSlocation'], copy=False), 'PBSlocation')
    rtt_wt_l, rtt_wt_r = _parse_pridict_location_column(
        pd.Series(df['RT_initial_location'], copy=False), 'RT_initial_location'
    )
    rtt_mut_l, rtt_mut_r = _parse_pridict_location_column(
        pd.Series(df['RT_mutated_location'], copy=False), 'RT_mutated_location'
    )

    rha_len = pd.Series(df['RTToverhang'], copy=False).astype('string').str.upper().str.len().astype(int)
    lha_len = pd.Series(np.where(
        type_del,
        rtt_mut_r - rtt_mut_l - rha_len,
        rtt_mut_r - rtt_mut_l - rha_len - edit_len,
    ), index=df.index).astype(int)
    lha_l = rtt_wt_l
    lha_r = rtt_wt_l + lha_len

    rha_wt_l = rtt_wt_r - rha_len
    rha_mut_r = rtt_mut_r

    # ---- Step 4: align WT and mut sequences on edit position ----
    aligned = pd.DataFrame(
        {
            'wt': wt_sequence,
            'mut': mut_sequence,
            'lha_r': lha_r,
            'edit_len': edit_len,
            'edit_type': edit_type,
        }
    ).apply(
        lambda row: align_wt_mut_sequences(
            row['wt'],
            row['mut'],
            int(row['lha_r']),
            edit_length=int(row['edit_len']),
            edit_type=int(row['edit_type']),
        ),
        axis=1,
        result_type='expand',
    )
    aligned.columns = ['wt_sequence', 'mut_sequence']
    wt_aligned = pd.Series(aligned['wt_sequence'], index=df.index)
    mut_aligned = pd.Series(aligned['mut_sequence'], index=df.index)
    coords = _shift_coords_after_wt_mut_align(
        index=df.index,
        type_ins=type_ins,
        type_del=type_del,
        edit_len=edit_len,
        lha_r=lha_r,
        protospacer_l=protospacer_l,
        protospacer_r=protospacer_r,
        pbs_l=pbs_l,
        pbs_r=pbs_r,
        rtt_l=rtt_wt_l,
        rtt_r=rtt_mut_r,
        lha_l=lha_l,
        rha_l=rha_wt_l,
        rha_r=rha_mut_r,
    )

    # ---- Step 5: assemble score/efficiency/fold fields ----
    spcas9_score = pd.Series(pd.to_numeric(df['deepcas9'], errors='coerce'), index=df.index)
    efficiency_column = next(
        (col for col in df.columns if col.lower().endswith('averageedited')),
        None,
    )
    if efficiency_column is None:
        raise ValueError("Could not find cell-line specific averageedited column in PRIDICT2 data.")
    editing_efficiency = pd.Series(pd.to_numeric(df[efficiency_column], errors='coerce'), index=df.index)
    if "testset_fold" in df.columns:
        original_fold = pd.to_numeric(df["testset_fold"], errors="coerce")
    else:
        original_fold = None

    # ---- Step 6: build and save standardized output ----
    output_df = _build_standardized_output_df(
        pd.Series(df['group_id'], index=df.index), type_sub, type_ins, type_del, edit_len,
        wt_aligned, mut_aligned, coords['protospacer_l'], coords['protospacer_r'],
        coords['pbs_l'], coords['pbs_r'], coords['rtt_l'], coords['rtt_r'],
        coords['lha_l'], coords['lha_r'], coords['rha_l'], coords['rha_r'],
        spcas9_score, editing_efficiency, original_fold)

    output_df = _attach_pridict_outcome_distribution(
        output_df,
        df,
        edited_column=efficiency_column,
    )

    output_path = DATA_ROOT / 'standardized' / 'pridict2' / dataset / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_path, index=False)
    logger.info("Saved standardized PRIDICT2 data: %s", output_path)

def _pridict2_trip_endo_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Map PRIDICT2 TRIP barcode rows to hg38 integration coordinates."""
    rows: list[dict[str, Any]] = []
    has_chr = "chromosome" in data.columns
    has_pos = "position" in data.columns
    has_barcode = "barcode" in data.columns
    for idx in data.index:
        barcode = str(data.at[idx, "barcode"]) if has_barcode else pd.NA
        chrom = data.at[idx, "chromosome"] if has_chr else pd.NA
        pos = data.at[idx, "position"] if has_pos else pd.NA
        if pd.isna(chrom) or pd.isna(pos):
            rows.append(
                {
                    "endo_genome_build": pd.NA,
                    "endo_chr": pd.NA,
                    "endo_start": pd.NA,
                    "endo_end": pd.NA,
                    "endo_strand": pd.NA,
                    "endo_coord_ref": pd.NA,
                    "endo_coord_source": pd.NA,
                    "endo_locus_id": barcode,
                }
            )
            continue
        # Published TRIP positions are 1-based hg38 (bowtie2 / ePRIDICT position_hg38).
        pos_1 = int(pos)
        rows.append(
            {
                "endo_genome_build": "hg38",
                "endo_chr": str(chrom),
                "endo_start": pos_1 - 1,
                "endo_end": pos_1,
                "endo_strand": pd.NA,
                "endo_coord_ref": "trip_integration",
                "endo_coord_source": "pridict2 trip_analysis export",
                "endo_locus_id": barcode,
            }
        )
    return pd.DataFrame(rows, index=data.index)


def _standardize_pridict2_trip(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Partial standardization for PRIDICT2 TRIP endogenous chromatin survey.

    TRIP supplementary table 12 reports barcode, genomic coordinates, and editing
    efficiencies only. All integrations were edited with the same pegRNA
    (``TRIP_pegRNA_GtoC`` in supplementary table 5): a uniform 1-bp G→C substitution
    used to survey chromatin context (Mathis et al., Nat. Biotechnol. 2024).
    """
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if "PE_editing_efficiency" not in data.columns:
        raise ValueError(
            f"PRIDICT2 TRIP export for {cell_line}-{pe_system} "
            "is missing PE_editing_efficiency."
        )
    editing_efficiency = pd.to_numeric(
        data["PE_editing_efficiency"], errors="coerce"
    ).fillna(0.0)

    type_sub = pd.Series(True, index=data.index, dtype=bool)
    type_ins = pd.Series(False, index=data.index, dtype=bool)
    type_del = pd.Series(False, index=data.index, dtype=bool)
    edit_len = pd.Series(1.0, index=data.index, dtype=float)

    partial_df = pd.DataFrame(
        {
            "type_sub": type_sub,
            "type_ins": type_ins,
            "type_del": type_del,
            "edit_len": edit_len,
            "editing_efficiency": editing_efficiency.astype(float),
        }
    )
    partial_df = _attach_endo_coordinate_columns(
        partial_df,
        _pridict2_trip_endo_coordinates(data),
    )
    _write_partial_standardized_output(
        partial_df,
        study="pridict2",
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_pridict_scaffold_assignments
    return [row for row in build_pridict_scaffold_assignments() if row.study == "pridict2"]


register_study(StudyPipeline(
    key="pridict2",
    exporters=(
        _export_pridict2_library_diverse_datasheets,
        _export_pridict2_endogenous_datasheets,
    ),
    standardizers={
        "library_diverse": _standardize_pridict2_library_diverse,
        "library_diverse_invivo": _standardize_pridict2_library_diverse,
        "trip_analysis": _standardize_pridict2_trip,
    },
    scaffold_assignments=_scaffold_assignments,
))

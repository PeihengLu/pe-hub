from __future__ import annotations

import logging
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
from ..pipeline.endo import (
    endo_coordinate_frame_from_loci,
    expand_endogenous_frame,
    load_pridict1_library2_genomic_loci,
    reference_windows_for_keys,
)
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
    _correction_type_to_flags,
    _parse_pridict_location_column,
)

def _export_pridict1_datasheets() -> None:
    """
    Export the PRIDICT1 datasheets
    """
    # starting from restoring the pridict1 data into one file
    # Load the three parts of the PRIDICT data
    part1 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part1.csv')
    part2 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part2.csv')
    part3 = pd.read_csv(DATA_ROOT / 'raw' / 'pridict1' / 'pridict1_library1_part3.csv')

    # Concatenate the parts back into a single DataFrame
    restored_data = pd.concat([part1, part2, part3], ignore_index=True)
    library1_pe_system = 'pe2'
    library1_cell_line = 'hek293t'

    # Save the restored data as one exported datasheet for standardization.
    output_path = (
        DATA_ROOT / 'exported' / 'pridict1' / 'library1' /
        f'{library1_cell_line}-{library1_pe_system}.csv'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    restored_data.to_csv(output_path, index=False)
    logger.info(f"Saved restored PRIDICT1 library1 data to {output_path}")


# (dataset folder, cell_line, pe_system, source efficiency column)
_PRIDICT1_LIBRARY2_EXPORTS: tuple[tuple[str, str, str, str], ...] = (
    ("library2", "hek293t", "pe2", "HEKOpti-Scaffold_PE2_averageedited"),
    ("library2", "hek293tmlh1dn", "pe2", "HEKOpti-Scaffold_PE2-dnMLH1_averageedited"),
    ("library2", "u2os", "pe2", "U2OS_PE2_averageedited"),
    ("library2", "u2osmlh1dn", "pe2", "U2OS_PE2-dnMLH1_averageedited"),
    ("library2", "u2os", "pemax", "U2OS_Pemax_averageedited"),
    ("library2", "u2osmlh1dn", "pemax", "U2OS_Pemax-dnMLH1_averageedited"),
    ("library2", "k562", "pe2", "K562_PE2_averageedited"),
    ("library2", "k562mlh1dn", "pe2", "K562_PE2-dnMLH1_averageedited"),
    ("library2", "k562", "pemax", "K562_Pemax_averageedited"),
    ("library2", "k562mlh1dn", "pemax", "K562_Pemax-dnMLH1_averageedited"),
    ("library2-invivo", "liver_gfpplus", "pe2", "Liver-GFPplus_PE2Adeno_averageedited"),
)


def _export_pridict1_library2_datasheets() -> None:
    """
    Export PRIDICT1 disease-focused subscreen (``pridict_library2.csv``).

    Source: supplementary disease-block subscreen (~1.9k pegRNAs) with editing
    measured across HEK293T, U2OS, and K562 in vitro (PE2 / PEmax; MLH1−/− as
    separate cell lines), plus GFP+ mouse liver in vivo (library2-invivo).
    """
    source_path = DATA_ROOT / "raw" / "pridict1" / "pridict_library2.csv"
    df = pd.read_csv(source_path)
    efficiency_columns = [col for col in df.columns if col.endswith("averageedited")]

    for dataset_name, cell_line, pe_system, efficiency_col in _PRIDICT1_LIBRARY2_EXPORTS:
        if efficiency_col not in df.columns:
            logger.warning(
                "PRIDICT1 library2 missing column %s; skipping export", efficiency_col
            )
            continue

        export_df = df.dropna(subset=[efficiency_col]).copy()
        other_efficiency_cols = [col for col in efficiency_columns if col != efficiency_col]
        export_df = export_df.drop(columns=other_efficiency_cols, errors="ignore")
        export_df = export_df.rename(columns={efficiency_col: "averageedited"})

        output_path = (
            DATA_ROOT
            / "exported"
            / "pridict1"
            / dataset_name
            / f"{cell_line}-{pe_system}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_path, index=False)
        logger.info(
            "Saved PRIDICT1 %s data (%s-%s): %s (%s rows)",
            dataset_name,
            cell_line,
            pe_system,
            output_path,
            len(export_df),
        )


_PRIDICT1_ENDOGENOUS_METADATA_PATH = (
    DATA_ROOT / "raw" / "pridict1" / "pridict1_endogenous_edit_metadata.csv"
)


def _assign_pridict1_endogenous_peg_roles(df: pd.DataFrame) -> pd.Series:
    """Map seq_id order within locus to library best / endo / worst pegRNA roles."""
    if not {"seq_id", "locus"}.issubset(df.columns):
        return pd.Series(pd.NA, index=df.index, dtype="string")

    seq_num = (
        df["seq_id"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )
    rank_in_locus = seq_num.groupby(df["locus"]).rank(method="first")
    return rank_in_locus.map({1: "best", 2: "endo", 3: "worst"}).astype("string")


def _load_pridict1_endogenous_edit_metadata() -> pd.DataFrame:
    """
    Per-pegRNA correction metadata derived from PRIDICT supplementary CRISPResso
    batch amplicons (uzh-dqbm-cmi/PRIDICT supplementary_files).
    """
    if not _PRIDICT1_ENDOGENOUS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing PRIDICT1 endogenous metadata: {_PRIDICT1_ENDOGENOUS_METADATA_PATH}"
        )
    meta = pd.read_csv(_PRIDICT1_ENDOGENOUS_METADATA_PATH, dtype={"locus": "string"})
    meta["peg_role"] = meta["peg_role"].astype("string").str.strip().str.lower()
    meta["correction_type"] = (
        meta["correction_type"].astype("string").str.strip().str.lower()
    )
    return meta


def _attach_pridict1_endogenous_correction_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Join Correction_Type / Correction_Length and boolean type flags onto export rows."""
    out = df.copy()
    out["peg_role"] = _assign_pridict1_endogenous_peg_roles(out)
    meta = _load_pridict1_endogenous_edit_metadata()
    out["locus_key"] = out["locus"].astype("string")
    merged = out.merge(
        meta,
        left_on=["locus_key", "peg_role"],
        right_on=["locus", "peg_role"],
        how="left",
        suffixes=("", "_meta"),
    )
    if merged["correction_type"].isna().any():
        missing = merged.loc[merged["correction_type"].isna(), ["seq_id", "locus", "peg_role"]]
        raise ValueError(
            "PRIDICT1 endogenous rows missing correction metadata:\n"
            f"{missing.head().to_string()}"
        )

    correction_type = merged["correction_type"]
    merged["Correction_Type"] = correction_type.str.capitalize()
    merged["Correction_Length"] = pd.to_numeric(
        merged["correction_length"], errors="raise"
    )
    merged["type_sub"] = correction_type.eq("replacement")
    merged["type_ins"] = correction_type.eq("insertion")
    merged["type_del"] = correction_type.eq("deletion")
    return merged.drop(
        columns=["locus_key", "locus_meta", "correction_type", "correction_length"],
        errors="ignore",
    )


def _export_pridict1_endogenous_datasheets() -> None:
    """
    Export PRIDICT1 endogenous-locus validation (Mathis et al., Nat. Biotechnol. 2023).

    Source: ``raw/pridict1/pridict_endogenous.csv`` — arrayed pegRNAs at native
    genomic loci with measured editing in HEK293T and K562 (PE2 conditions in the
    original study). Not part of the lentiviral self-targeting HTS library.

    Edit types come from ``pridict1_endogenous_edit_metadata.csv`` (amplicon vs HDR
    in the authors' endogenous CRISPResso batch). Rows are keyed by locus and peg
    role (seq_id rank: best, endo, worst library pegRNA per site).
    """
    source_path = DATA_ROOT / "raw" / "pridict1" / "pridict_endogenous.csv"
    df = pd.read_csv(source_path)
    base_df = _attach_pridict1_endogenous_correction_fields(df)

    # Mathis et al. 2023 endogenous validation: PE2 + plasmid transfection in each line.
    exports = (
        ("hek293t", "pe2", "HEK293T_averageedited"),
        ("k562", "pe2", "K562_averageedited"),
    )
    for cell_line, pe_system, efficiency_col in exports:
        # Keep only the target cell line's primary endogenous efficiency column.
        export_df = base_df.copy()
        other_efficiency_cols = [
            col
            for col in export_df.columns
            if col.endswith("_averageedited") and col != efficiency_col
        ]
        export_df = export_df.drop(columns=other_efficiency_cols, errors="ignore")

        out_dir = DATA_ROOT / "exported" / "pridict1" / "endogenous"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{cell_line}-{pe_system}.csv"
        export_df.to_csv(output_path, index=False)
        logger.info(
            "Saved PRIDICT1 endogenous data (%s, kept primary column %s): %s (%s rows)",
            cell_line,
            efficiency_col,
            output_path,
            len(export_df),
        )


def _load_pridict1_library2_genomic_loci() -> dict[str, Any]:
    """Load hg38/mm39 anchors for PRIDICT1 library2-invivo Names."""
    return load_pridict1_library2_genomic_loci()


def _pridict1_library2_endo_coordinates(
    names: pd.Series,
    genes: pd.Series,
) -> pd.DataFrame:
    """Map PRIDICT1 library2-invivo Name values to curated genomic coordinates."""
    return endo_coordinate_frame_from_loci(
        names.astype(str),
        _load_pridict1_library2_genomic_loci().get("loci", {}),
        source="pridict1_library2_genomic_loci.json",
        fallback_locus_id=genes.astype(str),
    )

def _standardize_pridict1(
        data: Optional[pd.DataFrame], cell_line: str, pe_system: str, dataset: str) -> None:
    """
    Standardize PRIDICT1 library exports (library1, library2, library2-invivo)
    to the shared PE schema.
    """
    dataset = _normalize_name(dataset)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    input_name = f"{cell_line}-{pe_system}.csv"
    output_name = f"{cell_line}-{pe_system}.parquet"
    if data is None:
        data = pd.read_csv(DATA_ROOT / 'exported' / 'pridict1' / dataset / input_name)
    logger.info(
        "Standardizing PRIDICT1 dataset=%s cell_line=%s pe_system=%s rows=%s",
        dataset,
        cell_line,
        pe_system,
        len(data),
    )

    # PRIDICT target strand values are quoted in the source CSV ("'Fw'" / "'Rv'").
    # Keep the provided wide_* sequence orientation for both strands.
    # These sequences/locations are already internally consistent with PRIDICT's
    # pegRNA design output.
    # Step 
    df = data.copy()

    # ---- Step 1: Determine mutation type and filter invalid rows ----
    correction_type = pd.Series(df['Correction_Type'], copy=False).astype('string').str.strip().str.lower()
    type_sub = correction_type.eq('replacement')
    type_ins = correction_type.eq('insertion')
    type_del = correction_type.eq('deletion')
    unknown_mask = ~(type_sub | type_ins | type_del)
    if unknown_mask.any():
        unknown_values = df.loc[unknown_mask, 'Correction_Type'].astype(str).unique().tolist()[:5]
        raise ValueError(f"Unsupported Correction_Type values: {unknown_values}")

    # used for alignment, 0 for substitution, 1 for insertion, 2 for deletion
    edit_type = pd.Series(
        np.select([type_sub, type_ins, type_del], [0, 1, 2], default=-1),
        index=df.index,
    ).astype(int)
    edit_len = pd.Series(pd.to_numeric(df['Correction_Length'], errors='raise'), index=df.index).astype(int)

    # ---- Step 2: Compute protospacer and assign group IDs ----
    wt_sequence = pd.Series(df['wide_initial_target'], copy=False).astype('string').str.upper()
    mut_sequence = pd.Series(df['wide_mutated_target'], copy=False).astype('string').str.upper()

    protospacer_l, protospacer_r = _parse_pridict_location_column(
        pd.Series(df['protospacerlocation_only_initial'], copy=False), 'protospacerlocation_only_initial'
    )
    protospacer_bounds = pd.DataFrame(
        {"seq": wt_sequence, "l": protospacer_l, "r": protospacer_r},
        index=df.index,
    )
    df['protospacer'] = protospacer_bounds.apply(
        lambda row: (
            row["seq"][int(row["l"]):int(row["r"])]
            if isinstance(row["seq"], str)
            else ""
        ),
        axis=1,
    )
    df['group_id'] = df.groupby('protospacer').ngroup()

    # ---- Step 3: Compute PBS, RTT, LHA and RHA locations ----
    pbs_l, pbs_r = _parse_pridict_location_column(pd.Series(df['PBSlocation'], copy=False), 'PBSlocation')
    rtt_wt_l, rtt_wt_r = _parse_pridict_location_column(
        pd.Series(df['RT_initial_location'], copy=False), 'RT_initial_location'
    )
    rtt_mut_l, rtt_mut_r = _parse_pridict_location_column(
        pd.Series(df['RT_mutated_location'], copy=False), 'RT_mutated_location'
    )

    rha_len = pd.Series(df['RToverhang_seq'], copy=False).astype('string').str.upper().str.len().astype(int)
    lha_len = pd.Series(np.where(
        type_del,
        rtt_mut_r - rtt_mut_l - rha_len, 
        rtt_mut_r - rtt_mut_l - rha_len - edit_len,
    ), index=df.index).astype(int)
    lha_l = rtt_wt_l
    lha_r = rtt_wt_l + lha_len

    rha_wt_l = rtt_wt_r - rha_len
    rha_wt_r = rtt_wt_r
    rha_mut_r = rtt_mut_r

    # ---- Step 4: Align the wt and mut sequences ----
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

    # ---- Step 5: Concatenate spcas9 score and editing efficiency ----
    spcas9_score = pd.Series(pd.to_numeric(df['deepcas9'], errors='coerce'), index=df.index)
    average_edited = pd.Series(pd.to_numeric(df['averageedited'], errors='coerce'), index=df.index)
    if 'PE2df_percentageedited' in df.columns:
        pe2_edited = pd.Series(pd.to_numeric(df['PE2df_percentageedited'], errors='coerce'), index=df.index)
        editing_efficiency = pe2_edited.where(pe2_edited.notna(), average_edited)
    else:
        editing_efficiency = average_edited

    # ---- Step 6: Build output DataFrame ----
    output_df = _build_standardized_output_df(
        pd.Series(df['group_id'], index=df.index), type_sub, type_ins, type_del, edit_len,
        wt_aligned, mut_aligned, coords['protospacer_l'], coords['protospacer_r'],
        coords['pbs_l'], coords['pbs_r'], coords['rtt_l'], coords['rtt_r'],
        coords['lha_l'], coords['lha_r'], coords['rha_l'], coords['rha_r'],
        spcas9_score, editing_efficiency)

    output_df = _attach_pridict_outcome_distribution(output_df, df)

    if dataset == "library2_invivo" and {"Name", "Gene"}.issubset(df.columns):
        output_df = _attach_endo_coordinate_columns(
            output_df,
            _pridict1_library2_endo_coordinates(df["Name"], df["Gene"]),
        )
        loci = _load_pridict1_library2_genomic_loci().get("loci", {})
        windows = reference_windows_for_keys(loci, df["Name"].astype(str))
        windows.index = output_df.index
        output_df = expand_endogenous_frame(output_df, windows)

    output_path = DATA_ROOT / 'standardized' / 'pridict1' / dataset / f"{output_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    logger.info(f"Saved standardized PRIDICT1 data to {output_path}")

def _standardize_pridict1_endo(
    data: pd.DataFrame,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> None:
    """Partial standardization for PRIDICT1 endogenous validation pegRNAs."""
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = _normalize_name(dataset)

    if {"type_sub", "type_ins", "type_del"}.issubset(data.columns):
        type_sub = data["type_sub"].astype(bool)
        type_ins = data["type_ins"].astype(bool)
        type_del = data["type_del"].astype(bool)
    elif "Correction_Type" in data.columns:
        type_sub, type_ins, type_del = _correction_type_to_flags(data["Correction_Type"])
    else:
        enriched = _attach_pridict1_endogenous_correction_fields(data)
        type_sub = enriched["type_sub"].astype(bool)
        type_ins = enriched["type_ins"].astype(bool)
        type_del = enriched["type_del"].astype(bool)

    preferred_efficiency = {
        "hek293t": "HEK293T_averageedited",
        "k562": "K562_averageedited",
    }
    efficiency_col = preferred_efficiency.get(cell_line)
    if efficiency_col is None or efficiency_col not in data.columns:
        raise ValueError(
            f"PRIDICT1 endogenous export for {cell_line}-{pe_system} "
            f"is missing efficiency column {efficiency_col!r}."
        )
    editing_efficiency = pd.to_numeric(data[efficiency_col], errors="coerce").fillna(0.0)

    if "Correction_Length" in data.columns:
        edit_len = pd.to_numeric(data["Correction_Length"], errors="coerce")
    else:
        edit_len = pd.Series(np.nan, index=data.index, dtype=float)

    partial = pd.DataFrame(
        {
            "type_sub": type_sub,
            "type_ins": type_ins,
            "type_del": type_del,
            "edit_len": edit_len,
            "editing_efficiency": editing_efficiency.astype(float),
        }
    )
    partial = _attach_pridict_outcome_distribution(
        partial,
        data,
        edited_column=efficiency_col,
    )
    _write_partial_standardized_output(
        partial,
        study="pridict1",
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )


def _scaffold_assignments(data_root=None):
    from ..catalog.datasheets import build_pridict_scaffold_assignments
    return [row for row in build_pridict_scaffold_assignments() if row.study == "pridict1"]


register_study(StudyPipeline(
    key="pridict1",
    exporters=(
        _export_pridict1_datasheets,
        _export_pridict1_library2_datasheets,
        _export_pridict1_endogenous_datasheets,
    ),
    standardizers={
        "library1": _standardize_pridict1,
        "library2": _standardize_pridict1,
        "library2_invivo": _standardize_pridict1,
        "endogenous": _standardize_pridict1_endo,
    },
    scaffold_assignments=_scaffold_assignments,
))

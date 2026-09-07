"""Study-agnostic standardized schema helpers."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import shift_coords_after_indel_pad

from .names import _normalize_name

logger = logging.getLogger(__name__)

standard_pe_data_columns = [
    "group_id",
    "type_sub",
    "type_ins",
    "type_del",
    "edit_len",
    "wt_sequence",
    "mut_sequence",
    "protospacer_location_l",
    "protospacer_location_r",
    "pbs_location_l",
    "pbs_location_r",
    "rtt_location_l",
    "rtt_location_r",
    "lha_location_l",
    "lha_location_r",
    "rha_location_l",
    "rha_location_r",
    "spcas9_score",
    "editing_efficiency",
    "original_fold",
]

endo_standard_columns = [
    "endo_genome_build",
    "endo_chr",
    "endo_start",
    "endo_end",
    "endo_strand",
    "endo_coord_ref",
    "endo_coord_source",
    "endo_locus_id",
]

def _empty_endo_coordinate_frame(index: pd.Index) -> pd.DataFrame:
    """Return nullable endogenous coordinate columns for ``index``."""
    n = len(index)
    return pd.DataFrame(
        {
            "endo_genome_build": pd.Series([pd.NA] * n, index=index, dtype="string"),
            "endo_chr": pd.Series([pd.NA] * n, index=index, dtype="string"),
            "endo_start": pd.Series([pd.NA] * n, index=index, dtype="Int64"),
            "endo_end": pd.Series([pd.NA] * n, index=index, dtype="Int64"),
            "endo_strand": pd.Series([pd.NA] * n, index=index, dtype="Int64"),
            "endo_coord_ref": pd.Series([pd.NA] * n, index=index, dtype="string"),
            "endo_coord_source": pd.Series([pd.NA] * n, index=index, dtype="string"),
            "endo_locus_id": pd.Series([pd.NA] * n, index=index, dtype="string"),
        },
        index=index,
    )


def _attach_endo_coordinate_columns(
    output_df: pd.DataFrame,
    endo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Ensure endogenous coordinate columns exist; overwrite with ``endo_df`` when given."""
    out = output_df.copy()
    empty = _empty_endo_coordinate_frame(out.index)
    if endo_df is None:
        for column in endo_standard_columns:
            if column not in out.columns:
                out[column] = empty[column]
    else:
        aligned = endo_df.reindex(out.index)
        for column in endo_standard_columns:
            if column in aligned.columns:
                out[column] = aligned[column]
            elif column not in out.columns:
                out[column] = empty[column]

    obsolete = [
        column
        for column in out.columns
        if column.startswith("endo_") and column not in endo_standard_columns
    ]
    if obsolete:
        out = out.drop(columns=obsolete)
    return out


def _coerce_original_fold(
    original_fold: Optional[pd.Series | np.ndarray],
    *,
    length: int,
) -> pd.Series:
    """Return nullable float fold ids; all-NaN when the source has no split metadata."""
    if original_fold is None:
        return pd.Series(np.nan, index=range(length), dtype="Float64")
    return pd.to_numeric(pd.Series(original_fold, copy=False), errors="coerce").astype(
        "Float64"
    )


def _as_coord_series(value, index: pd.Index) -> pd.Series:
    if np.isscalar(value) or isinstance(value, (int, np.integer)):
        return pd.Series(int(value), index=index, dtype=int)
    return pd.Series(value, index=index)


def _shift_coords_after_wt_mut_align(
    *,
    index: pd.Index,
    type_ins: pd.Series | np.ndarray,
    type_del: pd.Series | np.ndarray,
    edit_len: pd.Series | np.ndarray,
    lha_r: pd.Series | np.ndarray,
    protospacer_l,
    protospacer_r,
    pbs_l,
    pbs_r,
    rtt_l,
    rtt_r,
    lha_l,
    rha_l,
    rha_r,
) -> dict[str, pd.Series]:
    """Shift coordinates 3' of the indel pad onto grown aligned WT/Mut sequences.

    Insertions pad WT, so WT-indexed intervals (spacer, PBS, nick/RTT start, LHA,
    RHA start) move. Deletions pad Mut, so Mut-indexed ends (RTT/RHA right) move.
    ``lha_r`` is the pad origin and is left unchanged.
    """
    type_ins = pd.Series(type_ins, index=index).astype(bool)
    type_del = pd.Series(type_del, index=index).astype(bool)
    edit_len = pd.Series(edit_len, index=index)
    lha_r = pd.Series(lha_r, index=index)

    def shift_wt(coords) -> pd.Series:
        return shift_coords_after_indel_pad(
            _as_coord_series(coords, index), lha_r, edit_len, type_ins
        )

    def shift_mut(coords) -> pd.Series:
        return shift_coords_after_indel_pad(
            _as_coord_series(coords, index), lha_r, edit_len, type_del
        )

    return {
        "protospacer_l": shift_wt(protospacer_l),
        "protospacer_r": shift_wt(protospacer_r),
        "pbs_l": shift_wt(pbs_l),
        "pbs_r": shift_wt(pbs_r),
        "rtt_l": shift_wt(rtt_l),
        "rtt_r": shift_mut(rtt_r),
        "lha_l": shift_wt(lha_l),
        "lha_r": lha_r.astype(int),
        "rha_l": shift_wt(rha_l),
        "rha_r": shift_mut(rha_r),
    }


def _build_standardized_output_df(
    group_id: pd.Series | np.ndarray, 
    type_sub: pd.Series | np.ndarray, type_ins: pd.Series | np.ndarray, type_del: pd.Series | np.ndarray, edit_len: pd.Series | np.ndarray, 
    wt_sequence: pd.Series | np.ndarray, mut_sequence: pd.Series | np.ndarray, 
    protospacer_location_l: int | np.ndarray | pd.Series, protospacer_location_r: int | np.ndarray | pd.Series, 
    pbs_location_l: pd.Series | np.ndarray, pbs_location_r: pd.Series | np.ndarray, 
    rtt_location_l: pd.Series | np.ndarray, rtt_location_r: pd.Series | np.ndarray, 
    lha_location_l: pd.Series | np.ndarray, lha_location_r: pd.Series | np.ndarray, 
    rha_location_l: pd.Series | np.ndarray, rha_location_r: pd.Series | np.ndarray, 
    spcas9_score: pd.Series | np.ndarray, editing_efficiency: pd.Series | np.ndarray, 
    original_fold: Optional[pd.Series | np.ndarray] = None) -> pd.DataFrame:
    """
    Standardize the data types in output dataframe
    
    Args:
        group_id: Series of group IDs
        type_sub: Series of boolean type_sub values
        type_ins: Series of boolean type_ins values
        type_del: Series of boolean type_del values
        edit_len: Series of edit lengths
        wt_sequence: Series of wild type sequences
        mut_sequence: Series of mutated sequences
        protospacer_location_l: int of protospacer location left
        protospacer_location_r: int of protospacer location right
        pbs_location_l: Series of PBS location left
        pbs_location_r: Series of PBS location right
        rtt_location_l: Series of RTT location left
        rtt_location_r: Series of RTT location right
        lha_location_l: Series of LHA location left
        lha_location_r: Series of LHA location right
        rha_location_l: Series of RHA location left
        rha_location_r: Series of RHA location right
        spcas9_score: Series of spcas9 scores
        editing_efficiency: Series of editing efficiencies
        original_fold: Series of author-provided fold ids (NaN when unknown)
    Returns:
        DataFrame with correct types
    """
    n_rows = len(group_id)
    output_df = pd.DataFrame({
        'group_id': group_id,
        'type_sub': type_sub,
        'type_ins': type_ins,
        'type_del': type_del,
        'edit_len': edit_len,
        'wt_sequence': wt_sequence,
        'mut_sequence': mut_sequence,
        'protospacer_location_l': protospacer_location_l,
        'protospacer_location_r': protospacer_location_r,
        'pbs_location_l': pbs_location_l,
        'pbs_location_r': pbs_location_r,
        'rtt_location_l': rtt_location_l,
        'rtt_location_r': rtt_location_r,
        'lha_location_l': lha_location_l,
        'lha_location_r': lha_location_r,
        'rha_location_l': rha_location_l,
        'rha_location_r': rha_location_r,
        'spcas9_score': spcas9_score,
        'editing_efficiency': editing_efficiency,
        'original_fold': _coerce_original_fold(original_fold, length=n_rows),
    })

    # String transformations
    output_df['wt_sequence'] = output_df['wt_sequence'].str.upper()
    output_df['mut_sequence'] = output_df['mut_sequence'].str.upper()

    # Batch type conversions
    bool_columns = ['type_sub', 'type_ins', 'type_del']
    int_columns = [
        'group_id', 'edit_len',
        'protospacer_location_l', 'protospacer_location_r',
        'pbs_location_l', 'pbs_location_r',
        'rtt_location_l', 'rtt_location_r',
        'lha_location_l', 'lha_location_r',
        'rha_location_l', 'rha_location_r',
    ]
    float_columns = ['spcas9_score', 'editing_efficiency', 'original_fold']

    output_df[bool_columns] = output_df[bool_columns].astype(bool)
    output_df[int_columns] = output_df[int_columns].astype(int)
    output_df[float_columns] = output_df[float_columns].astype(float)

    return output_df

def _drop_unmeasured_efficiency_rows(
    df: pd.DataFrame, *, label: str
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ``editing_efficiency`` label is missing.

    A blank or non-numeric efficiency cell means the edit was never measured in
    this cell line, which is not the same as an efficiency of zero. Such rows
    cannot be supervised on, and every downstream consumer coerces a missing
    label to ``0.0`` (see ``convert_data._safe_float_series``), which would
    quietly teach models that unmeasured edits are inefficient. Removing them
    here keeps the standardized parquet the single source of truth for "rows you
    can train on", and keeps the reason visible in the standardization log.

    Applied after the study-specific standardizers have finished, so it cannot
    desynchronize the positional/index-aligned column attachments they perform.
    """
    if "editing_efficiency" not in df.columns:
        return df, 0
    missing = pd.to_numeric(df["editing_efficiency"], errors="coerce").isna()
    n_missing = int(missing.sum())
    if not n_missing:
        return df, 0
    logger.warning(
        "Dropped %s of %s row(s) from %s: no editing_efficiency measurement.",
        n_missing,
        len(df),
        label,
    )
    return df.loc[~missing].reset_index(drop=True), n_missing

def _write_partial_standardized_output(
    partial_df: pd.DataFrame,
    *,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
) -> None:
    """Persist a filter-only standardized parquet (not valid for model training)."""
    output_path = (
        DATA_ROOT / "standardized" / study / dataset / f"{cell_line}-{pe_system}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_df.to_parquet(output_path, index=False)
    logger.info(
        "Saved partial standardized data for filter-only use: %s (%s rows)",
        output_path,
        len(partial_df),
    )

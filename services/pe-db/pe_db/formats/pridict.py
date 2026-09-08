"""Standardized → PRIDICT / PRIDICT2 converters."""
from __future__ import annotations

import logging
import os
from concurrent.futures import as_completed
from typing import Optional

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

from pe_common.sequence_utils import (
    normalize_target_dna,
    remove_padding,
    sanitize_dna_sequence,
    unpadded_coordinate,
)

from ..pipeline.endo import ENDO_SPACER_OFFSET

from .common import (
    ProgressCallback,
    _col_as_series,
    _edit_length_series,
    _label_series,
    _report_progress_milestone,
    _safe_float_series,
    _safe_int_series,
)
from .thermo import (
    _compute_pridict2_gc_features,
    _compute_pridict2_tm_features,
    _get_viennarna,
    _reverse_complement,
    _rna_mfe_or_zero,
)

logger = logging.getLogger(__name__)


def _format_location(left: int, right: int) -> str:
    """Render a half-open interval in PRIDICT's ``"[l, r]"`` string form.

    An inverted interval is clamped to an empty span so the vendor parsers do
    not crash. Callers should use :func:`_warn_on_inverted_intervals` to report
    how many rows were affected, since clamping hides broken geometry.
    """
    left, right = int(left), int(right)
    if right < left:
        right = left
    return f"[{left}, {right}]"


def _warn_on_inverted_intervals(label: str, left: pd.Series, right: pd.Series) -> None:
    """Report, once per column pair, how many rows have ``right < left``."""
    n_inverted = int((pd.Series(right).to_numpy() < pd.Series(left).to_numpy()).sum())
    if n_inverted:
        logger.warning(
            "%s has %s row(s) with an inverted interval; those spans were clamped "
            "to empty and the resulting sequences are unreliable.",
            label,
            n_inverted,
        )


def _pridict_indel_correction_length(df: pd.DataFrame) -> pd.Series:
    """Gap size vendor ``align_seqs`` should insert after pads are dropped.

    Stored ``edit_len`` counts padded-schema N's. After ``_pridict_author_frame``
    the true insertion/deletion is ``|len(mut) - len(wt)|``. Using the stored
    length when that disagrees leaves WT/Mut unequal and blows up PBS columns.
    """
    stored = _safe_int_series(_edit_length_series(df), default=0)
    type_ins = _col_as_series(df, "type_ins", False).astype(bool)
    type_del = _col_as_series(df, "type_del", False).astype(bool)
    delta = (
        _col_as_series(df, "wt_sequence", "").astype(str).str.len()
        - _col_as_series(df, "mut_sequence", "").astype(str).str.len()
    ).abs()
    return stored.where(~(type_ins | type_del), delta).astype(int)


def _resolve_correction_type(type_sub: bool, type_ins: bool, type_del: bool) -> str:
    if bool(type_del):
        return "Deletion"
    if bool(type_ins):
        return "Insertion"
    if bool(type_sub):
        return "Replacement"
    return "Replacement"


def _rtt_wt_right_bounds(df: pd.DataFrame) -> pd.Series:
    """Derive WT RT right bound from standardized rtt coords and edit metadata.

    Standardized parquet stores ``rtt_location_l`` as the WT RT start and
    ``rtt_location_r`` as the mutated RT end. PRIDICT2 also needs the WT RT end
    for ``RT_initial_location``, which differs from the mutated end on indels.
    """
    rtt_mut_r = _safe_int_series(_col_as_series(df, "rtt_location_r", 0))
    edit_len = _pridict_indel_correction_length(df)
    type_ins = _col_as_series(df, "type_ins", False).astype(bool)
    type_del = _col_as_series(df, "type_del", False).astype(bool)
    rtt_wt_r = rtt_mut_r.copy()
    rtt_wt_r = rtt_wt_r.where(~type_ins, rtt_mut_r - edit_len)
    rtt_wt_r = rtt_wt_r.where(~type_del, rtt_mut_r + edit_len)
    return rtt_wt_r


_WT_INDEXED_COORD_COLUMNS = (
    "protospacer_location_l",
    "protospacer_location_r",
    "pbs_location_l",
    "pbs_location_r",
    "rtt_location_l",
    "lha_location_l",
    "lha_location_r",
    "rha_location_l",
)
_MUT_INDEXED_COORD_COLUMNS = (
    "rtt_location_r",
    "rha_location_r",
)


def _pridict_author_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop alignment pads and map coordinates onto the unpadded author frame.

    PRIDICT's vendor preprocessor aligns WT/Mut itself. Feeding already-padded
    (and previously truncated) sequences would double-pad and replace N with A.
    """
    out = df.copy()
    wt = _col_as_series(out, "wt_sequence", "").astype(str)
    mut = _col_as_series(out, "mut_sequence", "").astype(str)
    out["wt_sequence"] = wt.map(remove_padding)
    out["mut_sequence"] = mut.map(remove_padding)

    def _map_coords(seq_series: pd.Series, columns: tuple[str, ...]) -> None:
        for column in columns:
            if column not in out.columns:
                continue
            coords = _safe_int_series(_col_as_series(out, column, 0))
            out[column] = [
                unpadded_coordinate(seq, int(index))
                for seq, index in zip(seq_series, coords)
            ]

    _map_coords(wt, _WT_INDEXED_COORD_COLUMNS)
    _map_coords(mut, _MUT_INDEXED_COORD_COLUMNS)
    return _crop_pridict_extra_upstream(out)


PRIDICT_AUTHOR_SPACER_OFFSET = 10


def _crop_pridict_extra_upstream(df: pd.DataFrame) -> pd.DataFrame:
    """Crop extra 5' genomic context so the spacer starts at the author offset.

    Endogenous standardized rows sit on a 200 bp window (spacer at 90). PRIDICT's
    vendor preprocessor expects the author-like frame (spacer at 10). 3' sequence
    including long insertions is kept.
    """
    if "protospacer_location_l" not in df.columns:
        return df
    spacer_l = _safe_int_series(_col_as_series(df, "protospacer_location_l", 0))
    extra = (spacer_l - PRIDICT_AUTHOR_SPACER_OFFSET).clip(lower=0)
    extra = extra.where(spacer_l.ge(ENDO_SPACER_OFFSET), other=0)
    if not bool(extra.gt(0).any()):
        return df
    out = df.copy()
    wt = out["wt_sequence"].astype(str).tolist()
    mut = out["mut_sequence"].astype(str).tolist()
    extras = extra.to_numpy()
    out["wt_sequence"] = [seq[int(shift):] if int(shift) else seq for seq, shift in zip(wt, extras)]
    out["mut_sequence"] = [seq[int(shift):] if int(shift) else seq for seq, shift in zip(mut, extras)]
    coord_columns = _WT_INDEXED_COORD_COLUMNS + _MUT_INDEXED_COORD_COLUMNS
    for column in coord_columns:
        if column not in out.columns:
            continue
        out[column] = (_safe_int_series(_col_as_series(out, column, 0)) - extra).clip(lower=0)
    return out


# Continuous columns normalized by PRIDICT2's MinMaxNormalizer (dataset.py).
PRIDICT2_NORMALIZER_COLUMNS = (
    "Correction_Length",
    "RToverhangmatches",
    "RToverhanglength",
    "RTlength",
    "PBSlength",
    "MFE_protospacer",
    "MFE_protospacer_scaffold",
    "MFE_extension",
    "MFE_extension_scaffold",
    "MFE_protospacer_extension_scaffold",
    "MFE_rt",
    "MFE_pbs",
    "RTmt",
    "RToverhangmt",
    "PBSmt",
    "protospacermt",
    "extensionmt",
    "original_base_mt",
    "edited_base_mt",
    "Tm1",
    "Tm2",
    "Tm2new",
    "Tm3",
    "Tm4",
    "TmD",
    "nGCcnt1",
    "nGCcnt2",
    "nGCcnt3",
    "fGCcont1",
    "fGCcont2",
    "fGCcont3",
)

# PE2 scaffold used by PRIDICT/PRIDICT2 author feature engineering (pegRNA design).
PRIDICT2_PE2_SCAFFOLD = (
    "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)


def _pridict2_mfe_parallel_min_rows() -> int:
    return int(os.getenv("PRIDICT2_MFE_PARALLEL_MIN_ROWS", "256"))


def _pridict2_mfe_worker_count() -> int:
    configured = os.getenv("PRIDICT2_MFE_WORKERS", "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, os.cpu_count() or 1)


def _pridict2_mfe_chunk_worker(
    chunk: list[tuple[str, str, dict[str, int]]],
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for wt, mut, seq_kwargs in chunk:
        results.append(_compute_pridict2_mfe_features(wt, mut, **seq_kwargs))
    return results


def _compute_pridict2_mfe_features_batch(
    payloads: list[tuple[str, str, dict[str, int]]],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[dict[str, float]]:
    if len(payloads) < _pridict2_mfe_parallel_min_rows():
        results: list[dict[str, float]] = []
        last_milestone = [-1]
        total = len(payloads)
        for index, (wt, mut, seq_kwargs) in enumerate(payloads, start=1):
            results.append(_compute_pridict2_mfe_features(wt, mut, **seq_kwargs))
            _report_progress_milestone(
                progress_callback,
                phase="Computing RNA MFE features",
                done=index,
                total=total,
                last_milestone=last_milestone,
            )
        return results

    workers = min(_pridict2_mfe_worker_count(), len(payloads))
    chunk_count = workers * 8
    chunk_size = max(1, (len(payloads) + chunk_count - 1) // chunk_count)
    chunks = [payloads[i:i + chunk_size] for i in range(0, len(payloads), chunk_size)]
    results: list[dict[str, float]] = []
    last_milestone = [-1]
    total = len(payloads)
    done = 0
    from pe_db.mfe_worker import pridict2_mfe_chunk_worker

    from ..process_pool import get_mfe_process_pool

    pool = get_mfe_process_pool()
    futures = {
        pool.submit(pridict2_mfe_chunk_worker, chunk): index
        for index, chunk in enumerate(chunks)
    }
    ordered_chunks: list[Optional[list[dict[str, float]]]] = [None] * len(chunks)
    for future in as_completed(futures):
        index = futures[future]
        ordered_chunks[index] = future.result()
        done += len(ordered_chunks[index])
        _report_progress_milestone(
            progress_callback,
            phase="Computing RNA MFE features",
            done=done,
            total=total,
            last_milestone=last_milestone,
        )
    for chunk_result in ordered_chunks:
        if chunk_result is not None:
            results.extend(chunk_result)
    return results


def _occurrences_substring(haystack: str, needle: str) -> int:
    count = 0
    start = 0
    while True:
        start = haystack.find(needle, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def _compute_pridict2_rtoverhangmatches(mut: str, *, rha_l: int, rha_r: int) -> float:
    rt_overhang = mut[rha_l:rha_r].upper()
    overhang_len = len(rt_overhang)
    if overhang_len == 0:
        return 0.0
    return float(
        _occurrences_substring(
            mut[rha_l:rha_l + overhang_len + 15],
            rt_overhang,
        )
    )


def _compute_pridict2_mfe_features(
    wt: str,
    mut: str,
    *,
    protospacer_l: int,
    protospacer_r: int,
    pbs_l: int,
    pbs_r: int,
    rtt_l: int,
    rtt_r: int,
) -> dict[str, float]:
    RNA = _get_viennarna()

    protospacer = ("G" + wt[protospacer_l:protospacer_r]).upper()
    pbs_rc = _reverse_complement(wt[pbs_l:pbs_r].upper())
    rt_rc = _reverse_complement(mut[rtt_l:rtt_r].upper())
    extension = rt_rc + pbs_rc
    protospacer_scaffold = protospacer + PRIDICT2_PE2_SCAFFOLD
    extension_scaffold = PRIDICT2_PE2_SCAFFOLD + extension
    protospacer_extension_scaffold = protospacer + extension_scaffold

    fold = RNA.fold
    return {
        "MFE_protospacer": _rna_mfe_or_zero(fold, protospacer),
        "MFE_protospacer_scaffold": _rna_mfe_or_zero(fold, protospacer_scaffold),
        "MFE_extension": _rna_mfe_or_zero(fold, extension),
        "MFE_extension_scaffold": _rna_mfe_or_zero(fold, extension_scaffold),
        "MFE_protospacer_extension_scaffold": _rna_mfe_or_zero(
            fold, protospacer_extension_scaffold
        ),
        "MFE_rt": _rna_mfe_or_zero(fold, rt_rc),
        "MFE_pbs": _rna_mfe_or_zero(fold, pbs_rc),
    }


def _compute_pridict2_wallace_mt_features(
    wt: str,
    mut: str,
    *,
    protospacer_l: int,
    protospacer_r: int,
    pbs_l: int,
    pbs_r: int,
    rtt_l: int,
    rtt_r: int,
    rha_l: int,
    rha_r: int,
    edit_pos: int,
) -> dict[str, float]:
    protospacer = ("G" + wt[protospacer_l:protospacer_r]).upper()
    pbs_rc = _reverse_complement(wt[pbs_l:pbs_r].upper())
    rt_rc = _reverse_complement(mut[rtt_l:rtt_r].upper())
    rt_overhang_rc = _reverse_complement(mut[rha_l:rha_r].upper())
    extension = rt_rc + pbs_rc

    original_base = wt[edit_pos:edit_pos + 1].upper() if edit_pos < len(wt) else "-"
    edited_base = mut[edit_pos:edit_pos + 1].upper() if edit_pos < len(mut) else "-"

    def _wallace(base: str) -> tuple[float, float]:
        if base in {"", "-", "N"}:
            return 0.0, 1.0
        return float(mt.Tm_Wallace(Seq(base))), 0.0

    original_base_mt, original_base_mt_nan = _wallace(original_base)
    edited_base_mt, edited_base_mt_nan = _wallace(edited_base)

    return {
        "protospacermt": float(mt.Tm_Wallace(Seq(protospacer))) if protospacer else 0.0,
        "extensionmt": float(mt.Tm_Wallace(Seq(extension))) if extension else 0.0,
        "RTmt": float(mt.Tm_Wallace(Seq(rt_rc))) if rt_rc else 0.0,
        "RToverhangmt": float(mt.Tm_Wallace(Seq(rt_overhang_rc))) if rt_overhang_rc else 0.0,
        "PBSmt": float(mt.Tm_Wallace(Seq(pbs_rc))) if pbs_rc else 0.0,
        "original_base_mt": original_base_mt,
        "edited_base_mt": edited_base_mt,
        "original_base_mt_nan": original_base_mt_nan,
        "edited_base_mt_nan": edited_base_mt_nan,
    }


def _enrich_pridict2_features(
    source: pd.DataFrame,
    out: pd.DataFrame,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Compute PRIDICT2 model features from the canonical standardized schema."""
    pbs_l = _safe_int_series(_col_as_series(source, "pbs_location_l", 0))
    pbs_r = _safe_int_series(_col_as_series(source, "pbs_location_r", 0))
    rtt_l = _safe_int_series(_col_as_series(source, "rtt_location_l", 0))
    rtt_r = _safe_int_series(_col_as_series(source, "rtt_location_r", 0))
    rha_l = _safe_int_series(_col_as_series(source, "rha_location_l", 0))
    rha_r = _safe_int_series(_col_as_series(source, "rha_location_r", 0))
    prot_r = _safe_int_series(_col_as_series(source, "protospacer_location_r", 0))
    prot_l = _safe_int_series(_col_as_series(source, "protospacer_location_l", 0))
    edit_len = _pridict_indel_correction_length(source)
    type_sub = _col_as_series(source, "type_sub", False).astype(bool)
    type_ins = _col_as_series(source, "type_ins", False).astype(bool)
    type_del = _col_as_series(source, "type_del", False).astype(bool)
    edit_pos = _safe_int_series(_col_as_series(source, "lha_location_r", 0))

    out["deepcas9"] = _safe_float_series(_col_as_series(source, "spcas9_score", 0.0), default=0.0)
    # Keep integer dtype: vendor sequence alignment uses Correction_Length as a repeat count.
    out["Correction_Length"] = edit_len.astype(int)
    out["PBSlength"] = (pbs_r - pbs_l).clip(lower=0).astype(float)
    out["RTlength"] = (rtt_r - rtt_l).clip(lower=0).astype(float)
    out["RToverhanglength"] = (rha_r - rha_l).clip(lower=0).astype(float)

    wt_series = _col_as_series(source, "wt_sequence", "").astype(str).map(normalize_target_dna)
    mut_series = _col_as_series(source, "mut_sequence", "").astype(str).map(normalize_target_dna)

    row_indices = list(source.index)
    mfe_payloads: list[tuple[str, str, dict[str, int]]] = []
    feature_rows: list[dict[str, float]] = []
    last_milestone = [-1]
    total_rows = len(row_indices)
    for row_number, row_idx in enumerate(row_indices, start=1):
        wt = wt_series.loc[row_idx]
        mut = mut_series.loc[row_idx]
        pbs_seq = sanitize_dna_sequence(mut[pbs_l.loc[row_idx]:pbs_r.loc[row_idx]], drop=True)
        rt_seq = sanitize_dna_sequence(mut[rtt_l.loc[row_idx]:rtt_r.loc[row_idx]], drop=True)
        seq_kwargs = dict(
            protospacer_l=int(prot_l.loc[row_idx]),
            protospacer_r=int(prot_r.loc[row_idx]),
            pbs_l=int(pbs_l.loc[row_idx]),
            pbs_r=int(pbs_r.loc[row_idx]),
            rtt_l=int(rtt_l.loc[row_idx]),
            rtt_r=int(rtt_r.loc[row_idx]),
        )
        feature_rows.append(
            {
                **_compute_pridict2_gc_features(pbs_seq, rt_seq),
                **_compute_pridict2_tm_features(
                    wt,
                    pbs_seq,
                    rt_seq,
                    protospacer_r=int(prot_r.loc[row_idx]),
                    edit_len=int(edit_len.loc[row_idx]),
                    type_sub=bool(type_sub.loc[row_idx]),
                    type_ins=bool(type_ins.loc[row_idx]),
                    type_del=bool(type_del.loc[row_idx]),
                ),
                **_compute_pridict2_wallace_mt_features(
                    wt,
                    mut,
                    **seq_kwargs,
                    rha_l=int(rha_l.loc[row_idx]),
                    rha_r=int(rha_r.loc[row_idx]),
                    edit_pos=int(edit_pos.loc[row_idx]),
                ),
                "RToverhangmatches": _compute_pridict2_rtoverhangmatches(
                    mut,
                    rha_l=int(rha_l.loc[row_idx]),
                    rha_r=int(rha_r.loc[row_idx]),
                ),
            }
        )
        mfe_payloads.append((wt, mut, seq_kwargs))
        _report_progress_milestone(
            progress_callback,
            phase="Computing thermodynamic features",
            done=row_number,
            total=total_rows,
            last_milestone=last_milestone,
        )

    for row_feats, mfe_feats in zip(
        feature_rows,
        _compute_pridict2_mfe_features_batch(mfe_payloads, progress_callback=progress_callback),
    ):
        row_feats.update(mfe_feats)

    feat_df = pd.DataFrame(feature_rows, index=row_indices)
    for col in feat_df.columns:
        out[col] = feat_df[col]

    missing = [col for col in PRIDICT2_NORMALIZER_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(
            "PRIDICT2 conversion is missing required model features after enrichment: "
            f"{missing}"
        )

    for colname in PRIDICT2_NORMALIZER_COLUMNS:
        if colname == "Correction_Length":
            out[colname] = _safe_int_series(out[colname], default=0)
        else:
            out[colname] = _safe_float_series(out[colname], default=0.0)

    for colname in ("original_base_mt_nan", "edited_base_mt_nan"):
        out[colname] = _safe_float_series(out[colname], default=0.0)

    return out


def standardized_to_pridict_dataframe(
    df: pd.DataFrame,
    *,
    sequence_id_prefix: str = "seq_",
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into PRIDICT/PRIDICT2-compatible dataframe."""
    df = _pridict_author_frame(df)
    out = pd.DataFrame(index=df.index)
    out["seq_id"] = [f"{sequence_id_prefix}{i}" for i in range(len(df))]
    out["wide_initial_target"] = _col_as_series(df, "wt_sequence", "").astype(str).map(normalize_target_dna)
    out["wide_mutated_target"] = _col_as_series(df, "mut_sequence", "").astype(str).map(normalize_target_dna)
    out["deepeditposition"] = _safe_int_series(_col_as_series(df, "lha_location_r", 0), default=0)
    out["deepeditposition_lst"] = out["deepeditposition"].map(lambda x: f"[{x}]")
    out["Correction_Type"] = [
        _resolve_correction_type(sub, ins, dele)
        for sub, ins, dele in zip(
            _col_as_series(df, "type_sub", False),
            _col_as_series(df, "type_ins", False),
            _col_as_series(df, "type_del", False),
        )
    ]
    out["Correction_Length"] = _pridict_indel_correction_length(df)
    protospacer_l = _col_as_series(df, "protospacer_location_l", 0)
    protospacer_r = _col_as_series(df, "protospacer_location_r", 0)
    _warn_on_inverted_intervals("protospacer_location", protospacer_l, protospacer_r)
    out["protospacerlocation_only_initial"] = [
        _format_location(l, r) for l, r in zip(protospacer_l, protospacer_r)
    ]
    pbs_l = _col_as_series(df, "pbs_location_l", 0)
    pbs_r = _col_as_series(df, "pbs_location_r", 0)
    _warn_on_inverted_intervals("pbs_location", pbs_l, pbs_r)
    out["PBSlocation"] = [_format_location(l, r) for l, r in zip(pbs_l, pbs_r)]
    rtt_wt_l = _safe_int_series(_col_as_series(df, "rtt_location_l", 0))
    rtt_mut_r = _safe_int_series(_col_as_series(df, "rtt_location_r", 0))
    rtt_wt_r = _rtt_wt_right_bounds(df)
    _warn_on_inverted_intervals("rtt_location (WT)", rtt_wt_l, rtt_wt_r)
    _warn_on_inverted_intervals("rtt_location (mutated)", rtt_wt_l, rtt_mut_r)
    out["RT_initial_location"] = [
        _format_location(l, r) for l, r in zip(rtt_wt_l, rtt_wt_r)
    ]
    out["RT_mutated_location"] = [
        _format_location(l, r) for l, r in zip(rtt_wt_l, rtt_mut_r)
    ]
    if "editing_efficiency" in df.columns:
        out["averageedited"] = _label_series(_col_as_series(df, "editing_efficiency", 0.0))
    elif "averageedited" in df.columns:
        out["averageedited"] = _label_series(_col_as_series(df, "averageedited", 0.0))
    # Prefer the preserved distribution trio when present so KL/CE targets are
    # mutually consistent (editing_efficiency may come from a PE2-only column).
    if {"averageedited", "averageunedited", "averageindel"}.issubset(df.columns):
        out["averageedited"] = _label_series(_col_as_series(df, "averageedited", 0.0))
        out["averageunedited"] = _safe_float_series(_col_as_series(df, "averageunedited", 0.0), default=0.0)
        out["averageindel"] = _safe_float_series(_col_as_series(df, "averageindel", 0.0), default=0.0)
    else:
        for optional_col in ("averageunedited", "averageindel"):
            if optional_col in df.columns:
                out[optional_col] = _safe_float_series(_col_as_series(df, optional_col, 0.0), default=0.0)
    return _enrich_pridict2_features(df, out, progress_callback=progress_callback)

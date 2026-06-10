"""Standardized-to-model format converters hosted in pe-db service."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Optional

import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

from pe_common.sequence_utils import sanitize_dna_sequence


STANDARDIZED_REQUIRED_COLUMNS = {
    "wt_sequence",
    "mut_sequence",
    "edit_len",
    "type_sub",
    "type_ins",
    "type_del",
    "protospacer_location_l",
    "protospacer_location_r",
    "pbs_location_l",
    "pbs_location_r",
    "rtt_location_l",
    "rtt_location_r",
    "lha_location_r",
}


def has_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    return set(required_columns).issubset(df.columns)


def is_standardized_dataframe(df: pd.DataFrame) -> bool:
    return has_columns(df, STANDARDIZED_REQUIRED_COLUMNS)


def _col_as_series(df: pd.DataFrame, colname: str, default: Any = 0) -> pd.Series:
    if colname in df.columns and isinstance(df[colname], pd.Series):
        return df[colname]
    return pd.Series(default, index=df.index)


def _edit_length_series(df: pd.DataFrame) -> pd.Series:
    """Read the edit length, accepting the canonical 'edit_len' or legacy 'edit_length'."""
    for colname in ("edit_len", "edit_length"):
        if colname in df.columns:
            return _col_as_series(df, colname, 0)
    return pd.Series(0, index=df.index)


def _safe_int_series(series: pd.Series, default: int = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric, index=series.index).fillna(default).astype(int)


def _safe_float_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric, index=series.index).fillna(default).astype(float)


def _format_location(left: int, right: int) -> str:
    return f"[{int(left)}, {int(right)}]"


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
    edit_len = _safe_int_series(_edit_length_series(df), default=0)
    type_ins = _col_as_series(df, "type_ins", False).astype(bool)
    type_del = _col_as_series(df, "type_del", False).astype(bool)
    rtt_wt_r = rtt_mut_r.copy()
    rtt_wt_r = rtt_wt_r.where(~type_ins, rtt_mut_r - edit_len)
    rtt_wt_r = rtt_wt_r.where(~type_del, rtt_mut_r + edit_len)
    return rtt_wt_r


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

_TM_FEATURE_NAMES = ("Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD")
_GC_FEATURE_NAMES = (
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
)
_MT_FEATURE_NAMES = (
    "protospacermt", "extensionmt", "RTmt", "RToverhangmt", "PBSmt",
    "original_base_mt", "edited_base_mt",
)

_viennarna = None

ProgressCallback = Callable[[str], None]


def _report_progress_milestone(
    progress_callback: Optional[ProgressCallback],
    *,
    phase: str,
    done: int,
    total: int,
    last_milestone: list[int],
) -> None:
    if progress_callback is None or total <= 0:
        return
    pct = int(100 * done / total)
    milestone = 100 if done >= total else (pct // 10) * 10
    if done >= total or milestone > last_milestone[0]:
        last_milestone[0] = milestone
        progress_callback(f"{phase}: {done}/{total} ({pct}%)")


def _get_viennarna():
    global _viennarna
    if _viennarna is None:
        import RNA

        _viennarna = RNA
    return _viennarna


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
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_result in pool.imap(_pridict2_mfe_chunk_worker, chunks):
            results.extend(chunk_result)
            done += len(chunk_result)
            _report_progress_milestone(
                progress_callback,
                phase="Computing RNA MFE features",
                done=done,
                total=total,
                last_milestone=last_milestone,
            )
    return results


def _reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def _gc_fraction_percent(seq: str) -> float:
    if not seq:
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    return 100.0 * gc_count / len(seq)


def _compute_pridict2_gc_features(pbs_seq: str, rt_seq: str) -> dict[str, float]:
    pbs = pbs_seq.upper()
    rt = rt_seq.upper()
    combined = pbs + rt
    n_gc_pbs = pbs.count("G") + pbs.count("C")
    n_gc_rt = rt.count("G") + rt.count("C")
    n_gc_combined = n_gc_pbs + n_gc_rt
    return {
        "nGCcnt1": float(n_gc_pbs),
        "nGCcnt2": float(n_gc_rt),
        "nGCcnt3": float(n_gc_combined),
        "fGCcont1": _gc_fraction_percent(pbs),
        "fGCcont2": _gc_fraction_percent(rt),
        "fGCcont3": _gc_fraction_percent(combined),
    }


def _compute_pridict2_tm_features(
    wt: str,
    pbs_seq: str,
    rt_seq: str,
    *,
    protospacer_r: int,
    edit_len: int,
    type_sub: bool,
    type_ins: bool,
    type_del: bool,
) -> dict[str, float]:
    pbs = pbs_seq.upper()
    rt = rt_seq.upper()
    n_nick = int(protospacer_r) - 3

    s_for_tm1 = _reverse_complement(pbs.replace("A", "U"))
    s_for_tm2 = wt[n_nick:n_nick + len(rt)]

    if type_sub:
        s_for_tm2new = wt[n_nick:n_nick + len(rt)]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt)])
    elif type_ins:
        s_for_tm2new = wt[n_nick:n_nick + len(rt) - edit_len]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt) - edit_len])
    elif type_del:
        s_for_tm2new = wt[n_nick:n_nick + len(rt) + edit_len]
        s_tm3_anti = _reverse_complement(wt[n_nick:n_nick + len(rt) + edit_len])
    else:
        s_for_tm2new = s_for_tm2
        s_tm3_anti = _reverse_complement(s_for_tm2)

    s_for_tm3 = [rt, s_tm3_anti]
    s_for_tm4 = [_reverse_complement(rt.replace("A", "U")), rt]

    tm1 = float(mt.Tm_NN(seq=Seq(s_for_tm1), nn_table=mt.R_DNA_NN1))
    tm2 = float(mt.Tm_NN(seq=Seq(s_for_tm2), nn_table=mt.DNA_NN3))
    tm2new = float(mt.Tm_NN(seq=Seq(s_for_tm2new), nn_table=mt.DNA_NN3))

    tm3 = 0.0
    for s_seq1, s_seq2 in zip(s_for_tm3[0], s_for_tm3[1]):
        try:
            tm3 = float(mt.Tm_NN(seq=s_seq1, c_seq=s_seq2, nn_table=mt.DNA_NN3))
        except ValueError:
            continue

    tm4 = float(mt.Tm_NN(seq=Seq(s_for_tm4[0]), nn_table=mt.R_DNA_NN1))
    tmD = tm3 - tm2
    return {
        "Tm1": tm1,
        "Tm2": tm2,
        "Tm2new": tm2new,
        "Tm3": tm3,
        "Tm4": tm4,
        "TmD": tmD,
    }


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

    return {
        "MFE_protospacer": float(RNA.fold(protospacer)[1]),
        "MFE_protospacer_scaffold": float(RNA.fold(protospacer_scaffold)[1]),
        "MFE_extension": float(RNA.fold(extension)[1]),
        "MFE_extension_scaffold": float(RNA.fold(extension_scaffold)[1]),
        "MFE_protospacer_extension_scaffold": float(RNA.fold(protospacer_extension_scaffold)[1]),
        "MFE_rt": float(RNA.fold(rt_rc)[1]),
        "MFE_pbs": float(RNA.fold(pbs_rc)[1]),
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


def _compute_deepprime_thermo_features(
    wt: str,
    pbs_seq: str,
    rt_seq: str,
    *,
    protospacer_l: int,
    protospacer_r: int,
    edit_len: int,
    type_sub: bool,
    type_ins: bool,
    type_del: bool,
) -> dict[str, float]:
    RNA = _get_viennarna()

    tm_feats = _compute_pridict2_tm_features(
        wt,
        pbs_seq,
        rt_seq,
        protospacer_r=protospacer_r,
        edit_len=edit_len,
        type_sub=type_sub,
        type_ins=type_ins,
        type_del=type_del,
    )
    gc_feats = _compute_pridict2_gc_features(pbs_seq, rt_seq)
    guide_seq = ("G" + wt[protospacer_l:protospacer_r]).upper()
    mfe3_seq = _reverse_complement((pbs_seq + rt_seq).upper()) + "TTTTTT"
    return {
        **tm_feats,
        **gc_feats,
        "MFE3": float(RNA.fold(mfe3_seq)[1]),
        "MFE4": float(RNA.fold(guide_seq)[1]),
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
    edit_len = _safe_int_series(_edit_length_series(source), default=0)
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

    wt_series = _col_as_series(source, "wt_sequence", "").astype(str).map(sanitize_dna_sequence)
    mut_series = _col_as_series(source, "mut_sequence", "").astype(str).map(sanitize_dna_sequence)

    row_indices = list(source.index)
    mfe_payloads: list[tuple[str, str, dict[str, int]]] = []
    feature_rows: list[dict[str, float]] = []
    last_milestone = [-1]
    total_rows = len(row_indices)
    for row_number, row_idx in enumerate(row_indices, start=1):
        wt = wt_series.loc[row_idx]
        mut = mut_series.loc[row_idx]
        pbs_seq = mut[pbs_l.loc[row_idx]:pbs_r.loc[row_idx]]
        rt_seq = mut[rtt_l.loc[row_idx]:rtt_r.loc[row_idx]]
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
    out = pd.DataFrame(index=df.index)
    out["seq_id"] = [f"{sequence_id_prefix}{i}" for i in range(len(df))]
    out["wide_initial_target"] = _col_as_series(df, "wt_sequence", "").astype(str).map(sanitize_dna_sequence)
    out["wide_mutated_target"] = _col_as_series(df, "mut_sequence", "").astype(str).map(sanitize_dna_sequence)
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
    out["Correction_Length"] = _safe_int_series(_edit_length_series(df), default=0)
    out["protospacerlocation_only_initial"] = [
        _format_location(l, r)
        for l, r in zip(_col_as_series(df, "protospacer_location_l", 0), _col_as_series(df, "protospacer_location_r", 0))
    ]
    out["PBSlocation"] = [
        _format_location(l, r)
        for l, r in zip(_col_as_series(df, "pbs_location_l", 0), _col_as_series(df, "pbs_location_r", 0))
    ]
    rtt_wt_l = _safe_int_series(_col_as_series(df, "rtt_location_l", 0))
    rtt_mut_r = _safe_int_series(_col_as_series(df, "rtt_location_r", 0))
    rtt_wt_r = _rtt_wt_right_bounds(df)
    out["RT_initial_location"] = [
        _format_location(l, r) for l, r in zip(rtt_wt_l, rtt_wt_r)
    ]
    out["RT_mutated_location"] = [
        _format_location(l, r) for l, r in zip(rtt_wt_l, rtt_mut_r)
    ]
    if "editing_efficiency" in df.columns:
        out["averageedited"] = _safe_float_series(_col_as_series(df, "editing_efficiency", 0.0), default=0.0)
    elif "averageedited" in df.columns:
        out["averageedited"] = _safe_float_series(_col_as_series(df, "averageedited", 0.0), default=0.0)
    for optional_col in ("averageunedited", "averageindel"):
        if optional_col in df.columns:
            out[optional_col] = _safe_float_series(_col_as_series(df, optional_col, 0.0), default=0.0)
    return _enrich_pridict2_features(df, out, progress_callback=progress_callback)


def standardized_to_deepprime_dataframe(
    df: pd.DataFrame,
    *,
    spcas9_column: str = "spcas9_score",
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into DeepPrime feature dataframe."""
    wt_series = _col_as_series(df, "wt_sequence", "").astype(str).map(sanitize_dna_sequence).to_numpy()
    mut_series = _col_as_series(df, "mut_sequence", "").astype(str).map(sanitize_dna_sequence).to_numpy()
    protospacer_l_series = _safe_int_series(_col_as_series(df, "protospacer_location_l", 0)).to_numpy()
    protospacer_r_series = _safe_int_series(_col_as_series(df, "protospacer_location_r", 0)).to_numpy()
    pbs_l_series = _safe_int_series(_col_as_series(df, "pbs_location_l", 0)).to_numpy()
    pbs_r_series = _safe_int_series(_col_as_series(df, "pbs_location_r", 0)).to_numpy()
    rtt_l_series = _safe_int_series(_col_as_series(df, "rtt_location_l", 0)).to_numpy()
    rtt_r_series = _safe_int_series(_col_as_series(df, "rtt_location_r", 0)).to_numpy()
    lha_r_series = _safe_int_series(_col_as_series(df, "lha_location_r", 0)).to_numpy()
    rha_l_series = _safe_int_series(_col_as_series(df, "rha_location_l", 0)).to_numpy()
    rha_r_series = _safe_int_series(_col_as_series(df, "rha_location_r", 0)).to_numpy()
    edit_len_series = _safe_int_series(_edit_length_series(df)).to_numpy()
    type_sub_series = _col_as_series(df, "type_sub", False).astype(bool).to_numpy()
    type_ins_series = _col_as_series(df, "type_ins", False).astype(bool).to_numpy()
    type_del_series = _col_as_series(df, "type_del", False).astype(bool).to_numpy()
    spcas9_series = _safe_float_series(_col_as_series(df, spcas9_column, 0.0), default=0.0).to_numpy()
    efficiency_series = (
        _safe_float_series(_col_as_series(df, "editing_efficiency", 0.0), default=0.0).to_numpy()
        if "editing_efficiency" in df.columns
        else None
    )

    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    last_milestone = [-1]
    for i in range(total_rows):
        wt = str(wt_series[i])
        mut = str(mut_series[i])
        protospacer_l = int(protospacer_l_series[i])
        protospacer_r = int(protospacer_r_series[i])
        pbs_l = int(pbs_l_series[i])
        pbs_r = int(pbs_r_series[i])
        rtt_l = int(rtt_l_series[i])
        rtt_r = int(rtt_r_series[i])
        lha_r = int(lha_r_series[i])
        rha_l = int(rha_l_series[i])
        rha_r = int(rha_r_series[i])
        edit_len = int(edit_len_series[i])

        pbs_seq = sanitize_dna_sequence(mut[pbs_l:pbs_r], drop=True)
        rtt_seq = sanitize_dna_sequence(mut[rtt_l:rtt_r], drop=True)
        pbs_len = max(1, len(pbs_seq))
        rt_len = max(1, len(rtt_seq))
        rt_pbs_len = pbs_len + rt_len

        wt74 = wt[max(0, protospacer_l - 4): max(0, protospacer_l - 4) + 74]
        if len(wt74) < 74:
            wt74 = wt74 + ("N" * (74 - len(wt74)))
        edited74 = ("X" * max(0, 21 - pbs_len)) + (pbs_seq + rtt_seq) + ("X" * max(0, 53 - rt_len))
        edited74 = edited74[:74]
        if len(edited74) < 74:
            edited74 = edited74 + ("X" * (74 - len(edited74)))

        edit_pos = int(max(1, min(rt_len, (lha_r - rtt_l + 1))))
        rha_len = int(max(1, rha_r - rha_l))

        thermo = _compute_deepprime_thermo_features(
            wt,
            pbs_seq,
            rtt_seq,
            protospacer_l=protospacer_l,
            protospacer_r=protospacer_r,
            edit_len=edit_len,
            type_sub=bool(type_sub_series[i]),
            type_ins=bool(type_ins_series[i]),
            type_del=bool(type_del_series[i]),
        )
        row: dict[str, Any] = {
            "WT74_On": wt74,
            "Edited74_On": edited74,
            "PBSlen": pbs_len,
            "RTlen": rt_len,
            "RT-PBSlen": rt_pbs_len,
            "Edit_pos": edit_pos,
            "Edit_len": edit_len,
            "RHA_len": rha_len,
            "type_sub": int(bool(type_sub_series[i])),
            "type_ins": int(bool(type_ins_series[i])),
            "type_del": int(bool(type_del_series[i])),
            "DeepSpCas9_score": float(spcas9_series[i]),
            **thermo,
        }
        if efficiency_series is not None:
            row["Efficiency"] = float(efficiency_series[i])
        rows.append(row)
        _report_progress_milestone(
            progress_callback,
            phase="Converting DeepPrime features",
            done=i + 1,
            total=total_rows,
            last_milestone=last_milestone,
        )
    return pd.DataFrame(rows, index=df.index)


def standardized_to_oped_dataframe(
    df: pd.DataFrame,
    *,
    target_len: int = 47,
    protospacer_upstream_bases: int = 4,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into OPED sequence dataframe."""
    efficiency = _safe_float_series(_col_as_series(df, "editing_efficiency", 0.0), default=0.0).to_numpy()
    wt_series = _col_as_series(df, "wt_sequence", "").astype(str).str.upper().to_numpy()
    mut_series = _col_as_series(df, "mut_sequence", "").astype(str).str.upper().to_numpy()
    pbs_l = _safe_int_series(_col_as_series(df, "pbs_location_l", 0), default=0).to_numpy()
    pbs_r = _safe_int_series(_col_as_series(df, "pbs_location_r", 0), default=0).to_numpy()
    rtt_l = _safe_int_series(_col_as_series(df, "rtt_location_l", 0), default=0).to_numpy()
    rtt_r = _safe_int_series(_col_as_series(df, "rtt_location_r", 0), default=0).to_numpy()
    prot_l = _safe_int_series(_col_as_series(df, "protospacer_location_l", 0), default=0).to_numpy()

    records: list[dict[str, Any]] = []
    total_rows = len(wt_series)
    last_milestone = [-1]
    for row_pos, (wt, mut, pbs_l_i, pbs_r_i, rtt_l_i, rtt_r_i, prot_l_i) in enumerate(
        zip(wt_series, mut_series, pbs_l, pbs_r, rtt_l, rtt_r, prot_l)
    ):
        ref_chars = []
        for i in range(min(len(wt), len(mut))):
            wt_base = wt[i]
            mut_base = mut[i]
            if wt_base in {"A", "C", "G", "T"}:
                ref_chars.append(wt_base)
            elif mut_base in {"A", "C", "G", "T"}:
                ref_chars.append(mut_base)
            else:
                ref_chars.append("A")
        if len(wt) > len(mut):
            ref_chars.extend(base if base in {"A", "C", "G", "T"} else "A" for base in wt[len(mut):])
        elif len(mut) > len(wt):
            ref_chars.extend(base if base in {"A", "C", "G", "T"} else "A" for base in mut[len(wt):])
        ref_seq = "".join(ref_chars)

        target_start = max(0, int(prot_l_i) - int(protospacer_upstream_bases))
        target_end = target_start + target_len
        if target_end > len(ref_seq):
            target_start = max(0, len(ref_seq) - target_len)
            target_end = len(ref_seq)

        target = sanitize_dna_sequence(ref_seq[target_start:target_end])
        pbs_l_i = max(0, int(pbs_l_i))
        pbs_r_i = min(len(ref_seq), int(pbs_r_i))
        rtt_l_i = max(0, int(rtt_l_i))
        rtt_r_i = min(len(ref_seq), int(rtt_r_i))
        pbs_seq = sanitize_dna_sequence(ref_seq[pbs_l_i:pbs_r_i])
        rt_seq = sanitize_dna_sequence(ref_seq[rtt_l_i:rtt_r_i])
        if len(target) < target_len:
            target = target + ("A" * (target_len - len(target)))

        records.append(
            {
                "Target(47bp)": target,
                "PBS": pbs_seq,
                "RT": rt_seq,
                "Efficiency": float(efficiency[row_pos]),
            }
        )
        _report_progress_milestone(
            progress_callback,
            phase="Converting OPED sequences",
            done=row_pos + 1,
            total=total_rows,
            last_milestone=last_milestone,
        )

    return pd.DataFrame(records, index=df.index)


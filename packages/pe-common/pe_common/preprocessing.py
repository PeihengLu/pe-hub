"""Shared dataframe preprocessing for model wrappers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from .sequence_utils import sanitize_dna_sequence


STANDARDIZED_BASE_COLUMNS = {
    "wt_sequence",
    "mut_sequence",
    "edit_length",
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
    """Return True when all required columns exist."""
    return set(required_columns).issubset(df.columns)


def is_standardized_dataframe(df: pd.DataFrame) -> bool:
    """Detect the shared standardized PE schema."""
    return has_columns(df, STANDARDIZED_BASE_COLUMNS)


def _safe_int_series(series: pd.Series, default: int = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric, index=series.index).fillna(default).astype(int)


def _safe_float_series(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(numeric, index=series.index).fillna(default).astype(float)


def _col_as_series(df: pd.DataFrame, colname: str, default: Any = 0) -> pd.Series:
    if colname in df.columns:
        maybe_series = df[colname]
        if isinstance(maybe_series, pd.Series):
            return maybe_series
    return pd.Series(default, index=df.index)


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


def standardized_to_pridict_dataframe(
    df: pd.DataFrame,
    *,
    sequence_id_prefix: str = "seq_",
) -> pd.DataFrame:
    """Convert standardized schema into PRIDICT/PRIDICT2 dataframe fields."""
    out = pd.DataFrame(index=df.index)
    out["seq_id"] = [f"{sequence_id_prefix}{i}" for i in range(len(df))]
    out["wide_initial_target"] = df["wt_sequence"].astype(str).map(sanitize_dna_sequence)
    out["wide_mutated_target"] = df["mut_sequence"].astype(str).map(sanitize_dna_sequence)
    out["deepeditposition"] = _safe_int_series(_col_as_series(df, "lha_location_r", 0), default=0)
    out["deepeditposition_lst"] = out["deepeditposition"].map(lambda x: f"[{x}]")
    out["Correction_Type"] = [
        _resolve_correction_type(sub, ins, dele)
        for sub, ins, dele in zip(df["type_sub"], df["type_ins"], df["type_del"])
    ]
    out["Correction_Length"] = _safe_int_series(_col_as_series(df, "edit_length", 0), default=0)
    out["protospacerlocation_only_initial"] = [
        _format_location(l, r)
        for l, r in zip(df["protospacer_location_l"], df["protospacer_location_r"])
    ]
    out["PBSlocation"] = [
        _format_location(l, r)
        for l, r in zip(df["pbs_location_l"], df["pbs_location_r"])
    ]
    out["RT_initial_location"] = [
        _format_location(l, r)
        for l, r in zip(df["rtt_location_l"], df["rtt_location_r"])
    ]
    out["RT_mutated_location"] = [
        _format_location(l, r)
        for l, r in zip(df["rtt_location_l"], df["rtt_location_r"])
    ]

    if "editing_efficiency" in df.columns:
        out["averageedited"] = _safe_float_series(_col_as_series(df, "editing_efficiency", 0.0), default=0.0)
    elif "averageedited" in df.columns:
        out["averageedited"] = _safe_float_series(_col_as_series(df, "averageedited", 0.0), default=0.0)

    for optional_col in ("averageunedited", "averageindel"):
        if optional_col in df.columns:
            out[optional_col] = _safe_float_series(_col_as_series(df, optional_col, 0.0), default=0.0)

    return out


def standardized_to_deepprime_features(
    df: pd.DataFrame,
    *,
    spcas9_column: str = "spcas9_score",
) -> pd.DataFrame:
    """
    Convert standardized schema into DeepPrime feature dataframe.

    The standardized dataset does not carry all DeepPrime thermodynamic inputs,
    so placeholder values are filled with 0.0 for those fields.
    """
    wt_series = df["wt_sequence"].astype(str).map(sanitize_dna_sequence).to_numpy()
    mut_series = df["mut_sequence"].astype(str).map(sanitize_dna_sequence).to_numpy()
    protospacer_l_series = _safe_int_series(_col_as_series(df, "protospacer_location_l", 0)).to_numpy()
    pbs_l_series = _safe_int_series(_col_as_series(df, "pbs_location_l", 0)).to_numpy()
    pbs_r_series = _safe_int_series(_col_as_series(df, "pbs_location_r", 0)).to_numpy()
    rtt_l_series = _safe_int_series(_col_as_series(df, "rtt_location_l", 0)).to_numpy()
    rtt_r_series = _safe_int_series(_col_as_series(df, "rtt_location_r", 0)).to_numpy()
    lha_r_series = _safe_int_series(_col_as_series(df, "lha_location_r", 0)).to_numpy()
    rha_l_series = _safe_int_series(_col_as_series(df, "rha_location_l", 0)).to_numpy()
    rha_r_series = _safe_int_series(_col_as_series(df, "rha_location_r", 0)).to_numpy()
    edit_len_series = _safe_int_series(_col_as_series(df, "edit_length", 0)).to_numpy()
    type_sub_series = df["type_sub"].astype(bool).to_numpy()
    type_ins_series = df["type_ins"].astype(bool).to_numpy()
    type_del_series = df["type_del"].astype(bool).to_numpy()
    spcas9_series = _safe_float_series(
        _col_as_series(df, spcas9_column, 0.0),
        default=0.0,
    ).to_numpy()
    efficiency_series = (
        _safe_float_series(_col_as_series(df, "editing_efficiency", 0.0), default=0.0).to_numpy()
        if "editing_efficiency" in df.columns
        else None
    )

    rows: list[dict[str, Any]] = []
    for i in range(len(df)):
        wt = str(wt_series[i])
        mut = str(mut_series[i])
        protospacer_l = int(protospacer_l_series[i])
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

        n_gc_pbs = sum(base in {"G", "C"} for base in pbs_seq)
        n_gc_rt = sum(base in {"G", "C"} for base in rtt_seq)
        n_gc_rtpbs = n_gc_pbs + n_gc_rt
        edit_pos = int(max(1, min(rt_len, (lha_r - rtt_l + 1))))
        rha_len = int(max(1, rha_r - rha_l))

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
            "Tm1": 0.0,
            "Tm2": 0.0,
            "Tm2new": 0.0,
            "Tm3": 0.0,
            "Tm4": 0.0,
            "TmD": 0.0,
            "nGCcnt1": n_gc_pbs,
            "nGCcnt2": n_gc_rt,
            "nGCcnt3": n_gc_rtpbs,
            "fGCcont1": (100.0 * n_gc_pbs / pbs_len) if pbs_len else 0.0,
            "fGCcont2": (100.0 * n_gc_rt / rt_len) if rt_len else 0.0,
            "fGCcont3": (100.0 * n_gc_rtpbs / rt_pbs_len) if rt_pbs_len else 0.0,
            "MFE3": 0.0,
            "MFE4": 0.0,
            "DeepSpCas9_score": float(spcas9_series[i]),
        }
        if efficiency_series is not None:
            row["Efficiency"] = float(efficiency_series[i])
        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


def ensure_schema(
    df: pd.DataFrame,
    *,
    native_required: Iterable[str],
    converters: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Return dataframe in a target native schema using available converters.

    `converters` maps a name to a callable `(pd.DataFrame) -> pd.DataFrame`.
    """
    native_required_set = set(native_required)
    if native_required_set.issubset(df.columns):
        return df.copy()

    for converter in converters.values():
        converted = converter(df)
        if native_required_set.issubset(converted.columns):
            return converted

    missing = sorted(native_required_set.difference(df.columns))
    raise ValueError(f"Unsupported input schema. Missing required columns: {missing}")

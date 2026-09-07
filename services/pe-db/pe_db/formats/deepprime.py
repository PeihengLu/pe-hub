"""Standardized → DeepPrime converters."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from pe_common.sequence_utils import (
    normalize_target_dna,
    remove_padding,
    sanitize_dna_sequence,
    unpadded_coordinate,
)

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
)

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


def standardized_to_deepprime_dataframe(
    df: pd.DataFrame,
    *,
    spcas9_column: str = "spcas9_score",
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into DeepPrime feature dataframe."""
    wt_series = _col_as_series(df, "wt_sequence", "").astype(str).map(normalize_target_dna).to_numpy()
    mut_series = _col_as_series(df, "mut_sequence", "").astype(str).map(normalize_target_dna).to_numpy()
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
        _label_series(_col_as_series(df, "editing_efficiency", 0.0)).to_numpy()
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

        wt_unpadded = remove_padding(wt)
        spacer_start = unpadded_coordinate(wt, protospacer_l)
        # DeepPrime WT74 assumes 4 bp upstream of the 20-nt spacer. Hsu Lib-MMR
        # targets start at the spacer; left-pad unknown context instead of
        # clamping the crop to 0 (which frameshifts the spacer).
        pad_left = max(0, 4 - spacer_start)
        if pad_left:
            wt_unpadded = ("N" * pad_left) + wt_unpadded
            spacer_start += pad_left
        wt74_start = spacer_start - 4
        wt74 = wt_unpadded[wt74_start: wt74_start + 74]
        if len(wt74) < 74:
            wt74 = wt74 + ("N" * (74 - len(wt74)))
        edited74 = ("X" * max(0, 21 - pbs_len)) + (pbs_seq + rtt_seq) + ("X" * max(0, 53 - rt_len))
        edited74 = edited74[:74]
        if len(edited74) < 74:
            edited74 = edited74 + ("X" * (74 - len(edited74)))

        edit_pos = int(max(1, min(rt_len, (lha_r - rtt_l + 1))))
        rha_len = int(max(1, rha_r - rha_l))
        protospacer_l_unpadded = unpadded_coordinate(wt, protospacer_l) + pad_left
        protospacer_r_unpadded = unpadded_coordinate(wt, protospacer_r) + pad_left

        thermo = _compute_deepprime_thermo_features(
            wt_unpadded,
            pbs_seq,
            rtt_seq,
            protospacer_l=protospacer_l_unpadded,
            protospacer_r=protospacer_r_unpadded,
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

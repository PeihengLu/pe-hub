"""Standardized → OPED converters."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from pe_common.sequence_utils import sanitize_dna_sequence, unpadded_coordinate

from .common import (
    ProgressCallback,
    _col_as_series,
    _label_series,
    _report_progress_milestone,
    _safe_int_series,
)

def standardized_to_oped_dataframe(
    df: pd.DataFrame,
    *,
    target_len: int = 47,
    protospacer_upstream_bases: int = 4,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into OPED sequence dataframe.

    ``Target(47bp)`` is the unedited reporter window (WT with insertion-alignment
    pads dropped). Filling those pads from Mut would splice the insert into the
    WT 47-mer and frameshift Kim Wide-target. PBS is sliced from WT; the RT
    template is sliced from Mut.
    """
    efficiency = _label_series(_col_as_series(df, "editing_efficiency", 0.0)).to_numpy()
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
        # Unedited reporter: drop WT insertion pads rather than filling them
        # from Mut (that put the edit into Target and disagreed with Wide target).
        ref_seq = sanitize_dna_sequence(wt, drop=True)
        spacer_l = unpadded_coordinate(wt, int(prot_l_i))
        pad_left = max(0, int(protospacer_upstream_bases) - spacer_l)
        if pad_left:
            # OPED sanitize drops N; A is placeholder genomic context so the
            # 20-nt spacer stays at offset 4. Do not slide the window left when
            # the target is shorter than 47 bp — that frameshifts the spacer.
            ref_seq = ("A" * pad_left) + ref_seq
            spacer_l += pad_left
        target_start = spacer_l - int(protospacer_upstream_bases)
        target_end = target_start + target_len

        target = sanitize_dna_sequence(ref_seq[target_start:target_end])
        # PBS anneals to WT; the RT template is the edited (Mut) sequence.
        # Drop alignment pads rather than replacing them with A.
        pbs_l_i = max(0, int(pbs_l_i))
        pbs_r_i = min(len(wt), int(pbs_r_i))
        rtt_l_i = max(0, int(rtt_l_i))
        rtt_r_i = min(len(mut), int(rtt_r_i))
        pbs_seq = sanitize_dna_sequence(wt[pbs_l_i:pbs_r_i], drop=True)
        rt_seq = sanitize_dna_sequence(mut[rtt_l_i:rtt_r_i], drop=True)
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

"""Standardized → OptiPrime converters."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from pe_common.sequence_utils import normalize_target_dna, remove_padding, sanitize_dna_sequence

from .common import (
    ProgressCallback,
    _col_as_series,
    _label_series,
    _report_progress_milestone,
    _safe_int_series,
)

def standardized_to_optiprime_dataframe(
    df: pd.DataFrame,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert standardized schema into OptiPrime-compatible dataframe.

    OptiPrime (Hsu et al. 2026) requires columns: spacer, rtt, pbs,
    full_unedited, full_edited, scaffold_name, motif, cas9_type, pe_type,
    cas9_pam, time, edited_frac, indel_frac, weight.
    """
    from Bio.Seq import Seq as _Seq

    wt_series = _col_as_series(df, "wt_sequence", "").astype(str).map(normalize_target_dna)
    mut_series = _col_as_series(df, "mut_sequence", "").astype(str).map(normalize_target_dna)
    prot_l = _safe_int_series(_col_as_series(df, "protospacer_location_l", 0))
    prot_r = _safe_int_series(_col_as_series(df, "protospacer_location_r", 0))
    pbs_l = _safe_int_series(_col_as_series(df, "pbs_location_l", 0))
    pbs_r = _safe_int_series(_col_as_series(df, "pbs_location_r", 0))
    rtt_l = _safe_int_series(_col_as_series(df, "rtt_location_l", 0))
    rtt_r = _safe_int_series(_col_as_series(df, "rtt_location_r", 0))

    efficiency = _label_series(_col_as_series(df, "editing_efficiency", 0.0)).to_numpy()

    records: list[dict[str, Any]] = []
    total = len(df)
    last_milestone = [-1]
    for i in range(total):
        wt = str(wt_series.iloc[i])
        mut = str(mut_series.iloc[i])
        pl = int(prot_l.iloc[i])
        pr = int(prot_r.iloc[i])
        bl = int(pbs_l.iloc[i])
        br = int(pbs_r.iloc[i])
        rl = int(rtt_l.iloc[i])
        rr = int(rtt_r.iloc[i])

        protospacer = sanitize_dna_sequence(wt[pl:pr], drop=True)
        spacer_dna = protospacer
        if spacer_dna and spacer_dna[0] != "G":
            spacer_dna = "G" + spacer_dna

        pbs_dna = str(_Seq(sanitize_dna_sequence(wt[bl:br], drop=True)).reverse_complement())
        rtt_dna = str(_Seq(sanitize_dna_sequence(mut[rl:rr], drop=True)).reverse_complement())

        # OptiPrime uses RNA alphabet internally but the CSV input uses DNA
        spacer_rna = spacer_dna.replace("T", "U")
        pbs_rna = pbs_dna.replace("T", "U")
        rtt_rna = rtt_dna.replace("T", "U")

        # full_unedited / full_edited: region from 4bp upstream of protospacer
        # through the post-homology end, with alignment pads dropped.
        PS20_OFFSET = 4
        full_start = max(0, pl - PS20_OFFSET)
        full_u = remove_padding(wt[full_start:])
        full_e = remove_padding(mut[full_start:])

        records.append({
            "spacer": spacer_rna,
            "rtt": rtt_rna,
            "pbs": pbs_rna,
            "full_unedited": full_u,
            "full_edited": full_e,
            "scaffold_name": "BlpI_F+E",
            "motif": "tevoPreQ1",
            "cas9_type": "PEmax-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": "PE2",
            "time": 3.0,
            "edited_frac": float(efficiency[i]),
            "indel_frac": 0.0,
            "weight": 1.0,
            "Efficiency": float(efficiency[i]),
        })
        _report_progress_milestone(
            progress_callback,
            phase="Converting OptiPrime features",
            done=i + 1,
            total=total,
            last_milestone=last_milestone,
        )

    return pd.DataFrame(records, index=df.index)

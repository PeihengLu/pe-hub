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

# OptiPrime's coordinate frame: full_unedited/edited start 4 bp upstream of the
# 20 bp protospacer (Hsu DESIGN_PE / README).
PS20_OFFSET = 4

# Vendored OptiPrime scaffold_name keys (scripts/pe/pe_constants.SCAFFOLDS).
_SCAFFOLD_CONVENTIONAL = "SpCas9_OG"
_SCAFFOLD_OPTIMIZED = "GC_F+E"
_SCAFFOLD_OG_FE = "OG_F+E"
_SCAFFOLD_BLPI_FE = "BlpI_F+E"

# Trained group_factor names in vendor weights/model_*/log_rates/*.pkl.
_TRAINED_GROUPS = frozenset({
    "Kim_DLD1",
    "Liu_HEK293T",
    "Kim_MDA-MB-231",
    "Kim_HEK293T",
    "Schwank_U2OS",
    "Schwank_K562",
    "Kim_A549",
    "Liu_HeLa",
    "Schwank_HEK293T",
    "Kim_NIH3T3",
    "Kim_HeLa",
    "Kim_HCT116",
})

# Lab provenance is a weak prior for group_factors: Kim_HEK293T factors are
# hostile on DeepPE transfer, while Liu_HEK293T (Hsu's primary training context)
# ranks much better with the same PE2 assay flags. Prefer Liu → Kim → Schwank
# for the matched cell type when that group was trained.
_GROUP_LAB_PRIORITY = ("Liu", "Kim", "Schwank")

_CELL_LINE_VENDOR = {
    "hek293t": "HEK293T",
    "hela": "HeLa",
    "a549": "A549",
    "hap1": "HAP1",
    "k562": "K562",
    "k562mlh1dn": "K562",
    "u2os": "U2OS",
    "dld1": "DLD1",
    "mda_mb_231": "MDA-MB-231",
    "nih3t3": "NIH3T3",
    "hct116": "HCT116",
}


def _norm_key(value: Optional[str]) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _vendor_cell_type(cell_line: Optional[str]) -> str:
    key = _norm_key(cell_line)
    return _CELL_LINE_VENDOR.get(key, "HEK293T")


def _vendor_group(study: Optional[str], cell_line: Optional[str]) -> str:
    """Pick a trained group_factor key for this cell line.

    OptiPrime multiplies every ODE rate by ``exp(group_factor)``. Unknown names
    silently keep factor 0 (the old ``OptiPrime_HEK293T`` bug). Prefer Hsu's
    Liu_* factors when they exist for the cell type.
    """
    del study  # reserved for future lab-specific overrides
    cell = _vendor_cell_type(cell_line)
    for lab in _GROUP_LAB_PRIORITY:
        name = f"{lab}_{cell}"
        if name in _TRAINED_GROUPS:
            return name
    return "Liu_HEK293T"


def _pe_type_from_system(pe_system: Optional[str]) -> str:
    """OptiPrime MMR head is PE2 vs not-PE2; PE4 is the only trained non-PE2."""
    key = _norm_key(pe_system)
    if key.startswith("pe4") or "mlh1" in key:
        return "PE4"
    return "PE2"


def _assay_defaults(
    *,
    study: Optional[str],
    dataset: Optional[str],
    cell_line: Optional[str],
    pe_system: Optional[str],
) -> dict[str, Any]:
    """Study-conditioned OptiPrime CSV fields (mirrors vendor pe_datasets.process_*)."""
    study_key = _norm_key(study)
    dataset_key = _norm_key(dataset)
    cell = _vendor_cell_type(cell_line)
    pe_type = _pe_type_from_system(pe_system)
    pe_key = _norm_key(pe_system)

    # Hsu Lib-MMR / Lib-CV: BlpI_F+E epegRNA + PEmax, HEK 3 d / HeLa 5 d.
    if study_key == "optiprime":
        return {
            "scaffold_name": _SCAFFOLD_BLPI_FE,
            "motif": "tevoPreQ1",
            "cas9_type": "PEmax-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": pe_type,
            "time": 3.0 if cell == "HEK293T" else 5.0,
            "group": _vendor_group(study, cell_line),
        }

    # DeepPE (Y. Kim / H. Kim): conventional scaffold, PE2-Cas9, no epeg, ~3 d.
    if study_key == "deeppe":
        return {
            "scaffold_name": _SCAFFOLD_CONVENTIONAL,
            "motif": "none",
            "cas9_type": "PE2-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": "PE2",
            "time": 3.0,
            "group": _vendor_group(study, cell_line),
        }

    # DeepPrime / ClinVar (Kim): PE2 or PEmax, epeg when labelled, ~7–8 d.
    if study_key == "deepprime":
        is_max = "max" in pe_key
        is_epeg = "epeg" in pe_key or "epegrna" in pe_key
        is_nrch = "nrch" in pe_key
        return {
            "scaffold_name": _SCAFFOLD_CONVENTIONAL,
            "motif": "tevoPreQ1" if is_epeg else "none",
            "cas9_type": "PEmax-Cas9" if is_max else "PE2-Cas9",
            "cas9_pam": "SpNRCH" if is_nrch else "SpNGG",
            "pe_type": pe_type if pe_type in {"PE2", "PE4"} else "PE2",
            "time": 8.0 if "clinvar" in dataset_key else 7.0,
            "group": _vendor_group(study, cell_line),
        }

    # PRIDICT / PRIDICT2 (Schwank): optimized/Chen-like scaffold, ~7 d.
    if study_key in {"pridict1", "pridict2"}:
        is_max = "max" in pe_key
        return {
            "scaffold_name": _SCAFFOLD_OPTIMIZED,
            "motif": "none",
            "cas9_type": "PEmax-Cas9" if is_max else "PE2-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": "PE2",
            "time": 7.0,
            "group": _vendor_group(study, cell_line),
        }

    # Anzalone 2019 endogenous: conventional pegRNA, PE2-Cas9, no epeg.
    if study_key == "anzalone":
        return {
            "scaffold_name": _SCAFFOLD_CONVENTIONAL,
            "motif": "none",
            "cas9_type": "PE2-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": "PE2",
            "time": 3.0,
            "group": _vendor_group(study, cell_line),
        }

    # MinSePIE: conventional or near-OG_F+E scaffolds; PE2; epeg only when labelled.
    if study_key == "minsepie":
        is_epeg = "epeg" in pe_key or "epegrna" in pe_key
        scaffold = _SCAFFOLD_OG_FE if "codon" in dataset_key or "18nt" in dataset_key else _SCAFFOLD_CONVENTIONAL
        return {
            "scaffold_name": scaffold,
            "motif": "tevoPreQ1" if is_epeg else "none",
            "cas9_type": "PE2-Cas9",
            "cas9_pam": "SpNGG",
            "pe_type": "PE2",
            "time": 3.0,
            "group": _vendor_group(study, cell_line),
        }

    # Safe transfer default: PE2-Cas9 + conventional + Kim group factors.
    return {
        "scaffold_name": _SCAFFOLD_CONVENTIONAL,
        "motif": "none",
        "cas9_type": "PE2-Cas9",
        "cas9_pam": "SpNGG",
        "pe_type": pe_type,
        "time": 3.0,
        "group": _vendor_group(study, cell_line),
    }


def _ps20_aligned_targets(wt: str, mut: str, protospacer_l: int) -> tuple[str, str]:
    """Crop/pad so index 0 is 4 bp upstream of the protospacer.

    When the standardized window lacks upstream bases (Lib-MMR ``ps-pam-edit``
    starts at the spacer), left-pad with ``A`` — not ``N``, because
    ``remove_padding`` strips ``N`` and OptiPrime's alphabet is ACGT only.
    """
    pl = int(protospacer_l)
    missing = PS20_OFFSET - pl
    if missing > 0:
        pad = "A" * missing
        return pad + remove_padding(wt), pad + remove_padding(mut)
    start = pl - PS20_OFFSET
    return remove_padding(wt[start:]), remove_padding(mut[start:])


def _make_optiprime_spacer(full_unedited: str, protospacer: str) -> str:
    """Build spacer DNA using Hsu DESIGN_PE G21 rules (lowercase g = mismatch)."""
    if not protospacer:
        return ""
    if protospacer[0] == "G":
        return protospacer
    if len(full_unedited) >= PS20_OFFSET and full_unedited[PS20_OFFSET - 1] == "G":
        return full_unedited[PS20_OFFSET - 1 : PS20_OFFSET + 20]
    # Synthetic 5′ G not present in the target — mark mismatch for Plus1GMisMatch.
    return "g" + protospacer


def standardized_to_optiprime_dataframe(
    df: pd.DataFrame,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    study: Optional[str] = None,
    dataset: Optional[str] = None,
    cell_line: Optional[str] = None,
    pe_system: Optional[str] = None,
) -> pd.DataFrame:
    """Convert standardized schema into OptiPrime-compatible dataframe.

    OptiPrime (Hsu et al. 2026) requires columns: spacer, rtt, pbs,
    full_unedited, full_edited, scaffold_name, motif, cas9_type, pe_type,
    cas9_pam, time, edited_frac, indel_frac, weight, group.
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
    assay = _assay_defaults(
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )

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
        full_u, full_e = _ps20_aligned_targets(wt, mut, pl)
        spacer_dna = _make_optiprime_spacer(full_u, protospacer)

        pbs_dna = str(_Seq(sanitize_dna_sequence(wt[bl:br], drop=True)).reverse_complement())
        rtt_dna = str(_Seq(sanitize_dna_sequence(mut[rl:rr], drop=True)).reverse_complement())

        # Keep lowercase 'g' (Plus1GMisMatch); format_pe_df also maps T→U.
        if spacer_dna:
            head, tail = spacer_dna[0], spacer_dna[1:].upper().replace("T", "U")
            spacer_rna = ("g" if head == "g" else head.upper().replace("T", "U")) + tail
        else:
            spacer_rna = ""
        pbs_rna = pbs_dna.replace("T", "U")
        rtt_rna = rtt_dna.replace("T", "U")

        records.append({
            "spacer": spacer_rna,
            "rtt": rtt_rna,
            "pbs": pbs_rna,
            "full_unedited": full_u,
            "full_edited": full_e,
            "scaffold_name": assay["scaffold_name"],
            "motif": assay["motif"],
            "cas9_type": assay["cas9_type"],
            "cas9_pam": assay["cas9_pam"],
            "pe_type": assay["pe_type"],
            "time": float(assay["time"]),
            "group": assay["group"],
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

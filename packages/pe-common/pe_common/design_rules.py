"""Named pegRNA design-rule filters (Hsu / Anzalone practitioner criteria).

Hsu et al. (OptiPrime) argued that published PE libraries contain many pegRNAs
a practitioner would not order: extreme PBS/RTT lengths, a 3′ extension starting
with C, and homology arms far from the lengths used in real design. These
rulesets keep rows that match those design heuristics so evaluation can be
repeated on the well-designed subset.

Presets
=======
``optiprime`` / ``hsu``
    Hsu et al. library construction constraints that deposit consistently:
    PBS length 13; first nucleotide of the pegRNA RTT (3′ extension) is not C.
    The Methods also cite 3′ homology ``9+L`` / ``19+L`` with a C-avoiding
    length nudge; that clause is available as the atomic rule
    ``homology_hsu`` but is not part of this preset (deposited Lib-MMR / Lib-CV
    arms systematically deviate for multi-base edits and indels).

``anzalone``
    Anzalone et al. 2019 starting recommendations: PBS 10–16 nt, PBS GC 40–60%,
    RTT 10–16 nt, first RTT nucleotide not C.

Atomic names (``pbs_13``, ``rtt_not_c``, …) can be selected directly. Multiple
names are combined with AND.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

from .sequence_utils import reverse_complement, sanitize_dna_sequence

ATOMIC_DESIGN_RULES: frozenset[str] = frozenset(
    {
        "pbs_13",
        "pbs_10_16",
        "pbs_gc_40_60",
        "rtt_not_c",
        "rtt_10_16",
        "homology_hsu",
    }
)

DESIGN_RULESET_PRESETS: dict[str, tuple[str, ...]] = {
    "optiprime": ("pbs_13", "rtt_not_c"),
    "hsu": ("pbs_13", "rtt_not_c"),
    "anzalone": ("pbs_10_16", "pbs_gc_40_60", "rtt_10_16", "rtt_not_c"),
}

KNOWN_DESIGN_RULESETS: tuple[str, ...] = tuple(
    sorted(set(DESIGN_RULESET_PRESETS) | set(ATOMIC_DESIGN_RULES))
)

# Cap the C-nudge search so a pathological row cannot walk the whole sequence.
HSU_C_NUDGE_MAX_DELTA = 32


def expand_design_rulesets(names: Optional[Iterable[str]]) -> frozenset[str]:
    """Resolve preset aliases to atomic rule names. Empty/None → no rules."""
    if not names:
        return frozenset()
    resolved: set[str] = set()
    for raw in names:
        key = str(raw).strip().lower()
        if not key:
            continue
        if key in DESIGN_RULESET_PRESETS:
            resolved.update(DESIGN_RULESET_PRESETS[key])
            continue
        if key in ATOMIC_DESIGN_RULES:
            resolved.add(key)
            continue
        known = ", ".join(KNOWN_DESIGN_RULESETS)
        raise ValueError(f"Unknown design_ruleset {raw!r}; expected one of: {known}")
    return frozenset(resolved)


def hsu_c_nudge_homology_length(
    mut_sequence: str,
    rha_start: int,
    target: int,
    *,
    max_delta: int = HSU_C_NUDGE_MAX_DELTA,
) -> int:
    """Closest 3′ homology length to ``target`` whose distal DNA base is not G.

    Hsu et al. start at homology ``9+L`` / ``19+L``, then alter RTT/homology
    length so the first pegRNA RTT base is not C. That first pegRNA base is the
    reverse complement of the last DNA homology base, so avoiding C means
    avoiding G at the 3′ end of the DNA arm. On equal distance, prefer the
    shorter arm (``target-d`` before ``target+d``); that matches Hsu Lib-MMR
    single-SNP designs exactly.
    """
    if target <= 0:
        return 0
    mut = str(mut_sequence).upper().replace("U", "T")
    start = int(rha_start)
    if start < 0 or start >= len(mut):
        return 0
    max_len = len(mut) - start

    def ok(length: int) -> bool:
        if length < 1 or length > max_len:
            return False
        last = mut[start + length - 1]
        return last in "ACGT" and last != "G"

    if ok(target):
        return int(target)
    limit = min(int(max_delta), max(max_len, int(target)))
    for delta in range(1, limit + 1):
        for length in (target - delta, target + delta):
            if ok(length):
                return int(length)
    return 0


def apply_design_ruleset_mask(
    df: pd.DataFrame,
    rulesets: Optional[Iterable[str]],
) -> pd.Series:
    """Boolean mask of rows that satisfy every expanded design rule."""
    rules = expand_design_rulesets(rulesets)
    if not rules:
        return pd.Series(True, index=df.index)
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    geo = _pegRNA_geometry(df)
    mask = pd.Series(True, index=df.index)
    if "pbs_13" in rules:
        mask &= geo["pbs_len"] == 13
    if "pbs_10_16" in rules:
        mask &= geo["pbs_len"].between(10, 16)
    if "pbs_gc_40_60" in rules:
        mask &= geo["pbs_gc"].between(0.40, 0.60)
    if "rtt_not_c" in rules:
        mask &= geo["rtt_first"] != "C"
        mask &= geo["rtt_first"].ne("")
    if "rtt_10_16" in rules:
        mask &= geo["rtt_len"].between(10, 16)
    if "homology_hsu" in rules:
        mask &= geo["hsu_homology_target"] > 0
        mask &= geo["rha_len"] == geo["hsu_homology_nudged"]
        mask &= geo["hsu_homology_nudged"] > 0
    return mask.fillna(False)


def _col(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _pegRNA_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """PBS/RTT/homology lengths and pegRNA-orientation first RTT base."""
    wt = _col(df, "wt_sequence", "").astype(str)
    mut = _col(df, "mut_sequence", "").astype(str)
    pbs_l = pd.to_numeric(_col(df, "pbs_location_l", 0), errors="coerce").fillna(0).astype(int)
    pbs_r = pd.to_numeric(_col(df, "pbs_location_r", 0), errors="coerce").fillna(0).astype(int)
    rtt_l = pd.to_numeric(_col(df, "rtt_location_l", 0), errors="coerce").fillna(0).astype(int)
    rtt_r = pd.to_numeric(_col(df, "rtt_location_r", 0), errors="coerce").fillna(0).astype(int)
    rha_l = pd.to_numeric(_col(df, "rha_location_l", 0), errors="coerce").fillna(0).astype(int)
    rha_r = pd.to_numeric(_col(df, "rha_location_r", 0), errors="coerce").fillna(0).astype(int)
    edit_len = pd.to_numeric(_col(df, "edit_len", 0), errors="coerce").fillna(0).astype(int)
    type_sub = _col(df, "type_sub", False).astype(bool)
    type_ins = _col(df, "type_ins", False).astype(bool)
    type_del = _col(df, "type_del", False).astype(bool)

    pbs_len: list[int] = []
    pbs_gc: list[float] = []
    rtt_len: list[int] = []
    rtt_first: list[str] = []
    rha_len: list[int] = []
    hsu_target: list[int] = []
    hsu_nudged: list[int] = []

    for i in range(len(df)):
        pbs = sanitize_dna_sequence(str(wt.iloc[i])[int(pbs_l.iloc[i]) : int(pbs_r.iloc[i])], drop=True)
        mut_rtt_r = int(rtt_r.iloc[i])
        if bool(type_del.iloc[i]) and int(edit_len.iloc[i]) > 0:
            mut_rtt_r = max(int(rtt_l.iloc[i]), mut_rtt_r - int(edit_len.iloc[i]))
        rtt_dna = sanitize_dna_sequence(
            str(mut.iloc[i])[int(rtt_l.iloc[i]) : mut_rtt_r], drop=True
        )
        rtt_peg = reverse_complement(rtt_dna, mode="dna_to_dna") if rtt_dna else ""
        rha = sanitize_dna_sequence(
            str(mut.iloc[i])[int(rha_l.iloc[i]) : int(rha_r.iloc[i])], drop=True
        )
        length = len(pbs)
        pbs_len.append(length)
        gc = (pbs.count("G") + pbs.count("C")) / length if length else float("nan")
        pbs_gc.append(gc)
        rtt_len.append(len(rtt_peg))
        rtt_first.append(rtt_peg[:1])
        rha_len.append(len(rha))
        elen = int(edit_len.iloc[i])
        if bool(type_sub.iloc[i]):
            target = 9 + elen
        elif bool(type_ins.iloc[i]) or bool(type_del.iloc[i]):
            target = 19 + elen
        else:
            target = 0
        hsu_target.append(target)
        hsu_nudged.append(
            hsu_c_nudge_homology_length(str(mut.iloc[i]), int(rha_l.iloc[i]), target)
            if target > 0
            else 0
        )

    return pd.DataFrame(
        {
            "pbs_len": pbs_len,
            "pbs_gc": pbs_gc,
            "rtt_len": rtt_len,
            "rtt_first": rtt_first,
            "rha_len": rha_len,
            "hsu_homology_target": hsu_target,
            "hsu_homology_nudged": hsu_nudged,
        },
        index=df.index,
    )

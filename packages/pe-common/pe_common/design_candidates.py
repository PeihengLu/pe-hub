"""Enumerate practitioner-qualified pegRNA designs for a marked edit.

Users annotate the intended edit with ``(pre/after)`` markup on a target-strand
DNA sequence. This module finds SpCas9 ``NGG`` nick sites that can reach the
edit, enumerates PBS / RTT geometries for a named design policy (OptiPrime /
Hsu or Anzalone 2019), and returns standardized PE-core rows ready for model
scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd

from .design_rules import hsu_c_nudge_homology_length
from .sequence_utils import (
    align_wt_mut_sequences,
    normalize_target_dna,
    reverse_complement,
    shift_index_after_indel_pad,
)

EditType = Literal["sub", "ins", "del"]
DesignPolicy = Literal["optiprime", "anzalone"]

_EDIT_MARKUP_RE = re.compile(
    r"^(?P<left>[ACGTN]*)\((?P<pre>[ACGTN]*)/(?P<after>[ACGTN]*)\)(?P<right>[ACGTN]*)$",
    re.IGNORECASE,
)

SPACER_LEN = 20
NICK_OFFSET = 17  # nick sits between spacer bases 17 and 18 (0-based)
PAM_LEN = 3

# How far 3' of the nick an edit may start and still be PE-reachable.
DEFAULT_MAX_EDIT_DISTANCE = 40


@dataclass(frozen=True)
class ParsedEdit:
    """One edit parsed from ``(pre/after)`` markup on the target strand."""

    wt_sequence: str
    mut_sequence: str
    edit_type: EditType
    edit_len: int
    edit_pos: int  # 0-based start on the unpadded WT (ins/del) or WT (sub)
    pre: str
    after: str


@dataclass(frozen=True)
class PamSite:
    """SpCas9 NGG site on the target strand."""

    spacer_l: int
    spacer_r: int
    nick: int
    pam: str
    spacer: str


def parse_edit_markup(sequence: str) -> ParsedEdit:
    """Parse a single ``(pre/after)`` edit site into WT / Mut sequences."""
    if not str(sequence).strip():
        raise ValueError("Target sequence is empty")
    # Preserve edit markup; only strip whitespace before matching.
    compact = re.sub(r"\s+", "", str(sequence).strip()).upper().replace("U", "T")
    match = _EDIT_MARKUP_RE.match(compact)
    if match is None:
        raise ValueError(
            "Expected exactly one edit marked as (pre/after), e.g. "
            "ATCG(A/G)TCG, ATCG(/GGG)TCG, or ATCG(AAA/)TCG"
        )
    left = normalize_target_dna(match.group("left"))
    pre = normalize_target_dna(match.group("pre"))
    after = normalize_target_dna(match.group("after"))
    right = normalize_target_dna(match.group("right"))
    if any(base not in "ACGTN" for base in left + pre + after + right):
        raise ValueError("Target sequence must be DNA (A/C/G/T/N) plus one (pre/after) edit")
    # Reject ambiguous pads inside the annotated edit alleles.
    if any(base == "N" for base in pre + after):
        raise ValueError("Edit alleles in (pre/after) must be A/C/G/T (no N)")
    if "(" in left + right or ")" in left + right or "/" in left + right:
        raise ValueError("Only one (pre/after) edit site is supported")
    if not pre and not after:
        raise ValueError("Edit markup (pre/after) must change at least one base")

    edit_pos = len(left)
    if pre and after and len(pre) == len(after):
        edit_type: EditType = "sub"
        edit_len = len(pre)
        wt = left + pre + right
        mut = left + after + right
    elif not pre and after:
        edit_type = "ins"
        edit_len = len(after)
        wt = left + right
        mut = left + after + right
    elif pre and not after:
        edit_type = "del"
        edit_len = len(pre)
        wt = left + pre + right
        mut = left + right
    else:
        raise ValueError(
            "Unequal non-empty (pre/after) replacements are not supported; "
            "use a substitution of equal length, an insertion (/ins), or a deletion (del/)"
        )
    if edit_len < 1:
        raise ValueError("Edit length must be at least 1")
    return ParsedEdit(
        wt_sequence=wt,
        mut_sequence=mut,
        edit_type=edit_type,
        edit_len=edit_len,
        edit_pos=edit_pos,
        pre=pre,
        after=after,
    )


def find_spcas9_pam_sites(wt_sequence: str) -> list[PamSite]:
    """Return forward-strand SpCas9 NGG sites with a full 20-nt spacer."""
    wt = normalize_target_dna(wt_sequence)
    sites: list[PamSite] = []
    for spacer_l in range(0, max(0, len(wt) - SPACER_LEN - PAM_LEN + 1)):
        spacer_r = spacer_l + SPACER_LEN
        pam = wt[spacer_r : spacer_r + PAM_LEN]
        if len(pam) < PAM_LEN:
            break
        if pam[1:] != "GG" or pam[0] not in "ACGT":
            continue
        spacer = wt[spacer_l:spacer_r]
        if any(base not in "ACGT" for base in spacer):
            continue
        sites.append(
            PamSite(
                spacer_l=spacer_l,
                spacer_r=spacer_r,
                nick=spacer_l + NICK_OFFSET,
                pam=pam,
                spacer=spacer,
            )
        )
    return sites


def _pbs_gc_fraction(wt: str, nick: int, pbs_len: int) -> float:
    if pbs_len <= 0 or nick - pbs_len < 0:
        return float("nan")
    pbs = wt[nick - pbs_len : nick]
    if len(pbs) != pbs_len or any(base not in "ACGT" for base in pbs):
        return float("nan")
    return (pbs.count("G") + pbs.count("C")) / pbs_len


def _rtt_first_pegrna_base(mut: str, nick: int, rtt_r: int) -> str:
    """First base of the pegRNA RTT (RC of the DNA RTT 3′ end)."""
    if rtt_r <= nick or rtt_r > len(mut):
        return ""
    dna_rtt = mut[nick:rtt_r]
    if not dna_rtt or any(base not in "ACGT" for base in dna_rtt):
        return ""
    return reverse_complement(dna_rtt, mode="dna_to_dna")[:1]


def _edit_type_code(edit_type: EditType) -> int:
    return {"sub": 0, "ins": 1, "del": 2}[edit_type]


def _rha_start_unaligned(edit: ParsedEdit) -> int:
    if edit.edit_type == "del":
        return edit.edit_pos
    return edit.edit_pos + edit.edit_len


def _homology_span_needed(edit: ParsedEdit, nick: int) -> int:
    """Bases from nick through the end of the edit (exclusive of 3′ homology)."""
    return max(0, _rha_start_unaligned(edit) - nick)


def _policy_pbs_lengths(policy: DesignPolicy) -> Sequence[int]:
    if policy == "optiprime":
        return (13,)
    return tuple(range(10, 17))


def _policy_rtt_lengths(policy: DesignPolicy) -> Optional[Sequence[int]]:
    if policy == "anzalone":
        return tuple(range(10, 17))
    return None


def _build_standardized_row(
    edit: ParsedEdit,
    pam: PamSite,
    *,
    pbs_len: int,
    homology_len: int,
) -> Optional[dict]:
    """Build one standardized PE-core row, or None if geometry is invalid."""
    wt = edit.wt_sequence
    mut = edit.mut_sequence
    nick = pam.nick
    if nick - pbs_len < 0:
        return None
    if edit.edit_pos < nick:
        # Edit must start at or 3′ of the nick.
        return None

    rha_start = _rha_start_unaligned(edit)
    rtt_r = rha_start + homology_len
    if homology_len < 1 or rtt_r > len(mut):
        return None
    if any(base not in "ACGT" for base in wt[nick - pbs_len : nick]):
        return None
    if any(base not in "ACGT" for base in mut[nick:rtt_r]):
        return None

    first = _rtt_first_pegrna_base(mut, nick, rtt_r)
    if first == "C" or first == "":
        return None

    mut_type = _edit_type_code(edit.edit_type)
    wt_aligned, mut_aligned = align_wt_mut_sequences(
        wt, mut, edit.edit_pos, edit.edit_len, mut_type
    )

    type_ins = edit.edit_type == "ins"
    type_del = edit.edit_type == "del"

    def shift_wt(index: int) -> int:
        return shift_index_after_indel_pad(index, edit.edit_pos, edit.edit_len, type_ins)

    def shift_mut(index: int) -> int:
        return shift_index_after_indel_pad(index, edit.edit_pos, edit.edit_len, type_del)

    prot_l = shift_wt(pam.spacer_l)
    prot_r = shift_wt(pam.spacer_r)
    pbs_l = shift_wt(nick - pbs_len)
    pbs_r = shift_wt(nick)
    rtt_l = shift_wt(nick)
    lha_l = shift_wt(nick)
    lha_r = edit.edit_pos  # pad origin; unchanged by shift helpers
    rha_l = shift_wt(rha_start)
    rha_r = shift_mut(rtt_r)
    rtt_r_aligned = shift_mut(rtt_r)

    return {
        "group_id": hash(pam.spacer) % (2**31),
        "type_sub": edit.edit_type == "sub",
        "type_ins": type_ins,
        "type_del": type_del,
        "edit_len": int(edit.edit_len),
        "wt_sequence": wt_aligned,
        "mut_sequence": mut_aligned,
        "protospacer_location_l": int(prot_l),
        "protospacer_location_r": int(prot_r),
        "pbs_location_l": int(pbs_l),
        "pbs_location_r": int(pbs_r),
        "rtt_location_l": int(rtt_l),
        "rtt_location_r": int(rtt_r_aligned),
        "lha_location_l": int(lha_l),
        "lha_location_r": int(lha_r),
        "rha_location_l": int(rha_l),
        "rha_location_r": int(rha_r),
        "spcas9_score": float("nan"),
        "editing_efficiency": float("nan"),
        "original_fold": float("nan"),
        "pbs_len": int(pbs_len),
        "rtt_len": int(rtt_r - nick),
        "homology_len": int(homology_len),
        "spacer": pam.spacer,
        "pam": pam.pam,
        "nick": int(shift_wt(nick)),
    }


def enumerate_design_candidates(
    sequence: str,
    *,
    design_policy: DesignPolicy = "optiprime",
    max_edit_distance: int = DEFAULT_MAX_EDIT_DISTANCE,
) -> pd.DataFrame:
    """Enumerate pegRNA designs that satisfy ``design_policy`` for ``sequence``.

    Parameters
    ----------
    sequence:
        Target-strand DNA with one ``(pre/after)`` edit annotation.
    design_policy:
        ``optiprime`` (PBS=13, Hsu 9+L/19+L homology with C-nudge, RTT not starting
        with C) or ``anzalone`` (PBS 10–16 with 40–60% GC, RTT 10–16, RTT not C).
    max_edit_distance:
        Maximum nick→edit distance (bp) considered PE-reachable.
    """
    policy = str(design_policy).strip().lower()
    if policy in {"hsu", "optiprime"}:
        policy_key: DesignPolicy = "optiprime"
    elif policy == "anzalone":
        policy_key = "anzalone"
    else:
        raise ValueError(
            f"Unknown design_policy {design_policy!r}; expected 'optiprime' or 'anzalone'"
        )

    edit = parse_edit_markup(sequence)
    pam_sites = find_spcas9_pam_sites(edit.wt_sequence)
    rows: list[dict] = []

    for pam in pam_sites:
        distance = edit.edit_pos - pam.nick
        if distance < 0 or distance > int(max_edit_distance):
            continue
        span = _homology_span_needed(edit, pam.nick)
        if span < 0:
            continue

        for pbs_len in _policy_pbs_lengths(policy_key):
            if policy_key == "anzalone":
                gc = _pbs_gc_fraction(edit.wt_sequence, pam.nick, pbs_len)
                if not (0.40 <= gc <= 0.60):
                    continue

            homology_lengths: Iterable[int]
            if policy_key == "optiprime":
                target = (9 + edit.edit_len) if edit.edit_type == "sub" else (19 + edit.edit_len)
                # Homology starts at rha_start on Mut (unaligned).
                nudged = hsu_c_nudge_homology_length(
                    edit.mut_sequence,
                    _rha_start_unaligned(edit),
                    target,
                )
                if nudged < 1:
                    continue
                homology_lengths = (nudged,)
            else:
                rtt_lengths = _policy_rtt_lengths(policy_key) or ()
                homology_lengths = []
                for rtt_len in rtt_lengths:
                    homology = int(rtt_len) - int(span)
                    if homology >= 1:
                        homology_lengths.append(homology)
                homology_lengths = tuple(dict.fromkeys(homology_lengths))

            for homology_len in homology_lengths:
                row = _build_standardized_row(
                    edit,
                    pam,
                    pbs_len=int(pbs_len),
                    homology_len=int(homology_len),
                )
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)

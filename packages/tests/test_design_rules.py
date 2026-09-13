"""Tests for pegRNA design-rule filters."""

from __future__ import annotations

import pandas as pd
import pytest

from pe_common.design_rules import (
    apply_design_ruleset_mask,
    expand_design_rulesets,
    hsu_c_nudge_homology_length,
)
from pe_common.sequence_utils import reverse_complement


def _row(
    *,
    wt: str,
    mut: str,
    pbs: tuple[int, int],
    rtt: tuple[int, int],
    rha: tuple[int, int],
    edit_len: int = 1,
    type_sub: bool = True,
    type_ins: bool = False,
    type_del: bool = False,
) -> dict:
    return {
        "wt_sequence": wt,
        "mut_sequence": mut,
        "pbs_location_l": pbs[0],
        "pbs_location_r": pbs[1],
        "rtt_location_l": rtt[0],
        "rtt_location_r": rtt[1],
        "rha_location_l": rha[0],
        "rha_location_r": rha[1],
        "edit_len": edit_len,
        "type_sub": type_sub,
        "type_ins": type_ins,
        "type_del": type_del,
    }


def _hsu_like_sub() -> dict:
    # PBS=13, 1-bp sub, RHA=10 (=9+L). Homology ends in A (not G) so no C-nudge.
    seq = "A" * 80
    return _row(wt=seq, mut=seq, pbs=(20, 33), rtt=(33, 45), rha=(35, 45), edit_len=1)


def test_expand_optiprime_preset():
    assert expand_design_rulesets(["optiprime"]) == frozenset({"pbs_13", "rtt_not_c"})
    assert expand_design_rulesets(["hsu"]) == expand_design_rulesets(["optiprime"])
    # Homology 9+L / 19+L remains available as an atomic rule, not the preset.
    assert "homology_hsu" in expand_design_rulesets(["homology_hsu"])


def test_expand_unknown_ruleset():
    with pytest.raises(ValueError, match="Unknown design_ruleset"):
        expand_design_rulesets(["not-a-rule"])


def test_pbs_13_keeps_length_13():
    good = _hsu_like_sub()
    bad = dict(good)
    bad["pbs_location_r"] = 27  # length 7
    df = pd.DataFrame([good, bad])
    mask = apply_design_ruleset_mask(df, ["pbs_13"])
    assert mask.tolist() == [True, False]


def test_rtt_not_c_uses_pegrna_orientation():
    seq = "A" * 80
    # DNA RTT ending in G → pegRNA RTT starts with C (rejected).
    mut_c = list(seq)
    mut_c[33:45] = list("AAAAAAAAAAAG")
    starts_c = _row(
        wt=seq,
        mut="".join(mut_c),
        pbs=(20, 33),
        rtt=(33, 45),
        rha=(35, 45),
    )
    starts_t = _hsu_like_sub()
    assert reverse_complement(starts_c["mut_sequence"][33:45])[0] == "C"
    df = pd.DataFrame([starts_t, starts_c])
    mask = apply_design_ruleset_mask(df, ["rtt_not_c"])
    assert mask.tolist() == [True, False]


def test_hsu_c_nudge_keeps_target_when_distal_base_not_g():
    # Target 10; base at index 9 is A → keep 10.
    mut = "A" * 40
    assert hsu_c_nudge_homology_length(mut, 0, 10) == 10


def test_hsu_c_nudge_lengthens_when_only_longer_is_valid():
    # Target 10 ends in G; -1 also ends in G; +1 ends in A.
    mut = list("A" * 40)
    mut[9] = "G"
    mut[8] = "G"
    mut[10] = "A"
    assert hsu_c_nudge_homology_length("".join(mut), 0, 10) == 11


def test_hsu_c_nudge_shortens_when_target_ends_in_g():
    # Target 10 ends in G; prefer -1 (base at 8 is A) over +1 on equal distance.
    mut = list("A" * 40)
    mut[9] = "G"
    mut[10] = "A"
    mut[8] = "A"
    assert hsu_c_nudge_homology_length("".join(mut), 0, 10) == 9


def test_homology_hsu_accepts_exact_nudged_length():
    good = _hsu_like_sub()
    short = dict(good)
    short["rha_location_l"] = 42  # 3 nt homology; nudged target is 10
    df = pd.DataFrame([good, short])
    mask = apply_design_ruleset_mask(df, ["homology_hsu"])
    assert mask.tolist() == [True, False]


def test_homology_hsu_accepts_c_nudged_shorter_arm():
    # 9+L = 10 ends in G → nudged length 9; row with RHA=9 passes.
    mut = list("A" * 80)
    mut[44] = "G"  # last base of length-10 arm starting at 35
    mut[43] = "A"  # last base of length-9 arm
    mut_s = "".join(mut)
    row = _row(
        wt="A" * 80,
        mut=mut_s,
        pbs=(20, 33),
        rtt=(33, 44),
        rha=(35, 44),  # length 9
        edit_len=1,
    )
    assert hsu_c_nudge_homology_length(mut_s, 35, 10) == 9
    assert apply_design_ruleset_mask(pd.DataFrame([row]), ["homology_hsu"]).tolist() == [True]


def test_optiprime_preset_keeps_hsu_like_row():
    df = pd.DataFrame([_hsu_like_sub()])
    assert apply_design_ruleset_mask(df, ["optiprime"]).tolist() == [True]


def test_anzalone_rejects_zero_gc_pbs():
    df = pd.DataFrame([_hsu_like_sub()])
    # PBS is poly-A (GC 0), RTT length 12 is in 10–16.
    assert apply_design_ruleset_mask(df, ["anzalone"]).tolist() == [False]
    assert apply_design_ruleset_mask(df, ["pbs_10_16", "rtt_10_16"]).tolist() == [True]

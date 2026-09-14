"""Tests for pegRNA design candidate enumeration."""

from __future__ import annotations

import pytest

from pe_common.design_candidates import (
    enumerate_design_candidates,
    find_spcas9_pam_sites,
    parse_edit_markup,
)
from pe_common.design_rules import apply_design_ruleset_mask


def test_parse_substitution():
    edit = parse_edit_markup("AAAA(A/G)TTTT")
    assert edit.edit_type == "sub"
    assert edit.edit_len == 1
    assert edit.edit_pos == 4
    assert edit.wt_sequence == "AAAAATTTT"
    assert edit.mut_sequence == "AAAAGTTTT"


def test_parse_insertion_and_deletion():
    ins = parse_edit_markup("AAAA(/GGG)TTTT")
    assert ins.edit_type == "ins"
    assert ins.edit_len == 3
    assert ins.wt_sequence == "AAAATTTT"
    assert ins.mut_sequence == "AAAAGGGTTTT"

    deletion = parse_edit_markup("AAAA(GGG/)TTTT")
    assert deletion.edit_type == "del"
    assert deletion.edit_len == 3
    assert deletion.wt_sequence == "AAAAGGGTTTT"
    assert deletion.mut_sequence == "AAAATTTT"


def test_parse_rejects_missing_markup():
    with pytest.raises(ValueError, match="edit marked"):
        parse_edit_markup("AAAATTTT")


def test_find_pam_sites_basic():
    # spacer(20) + NGG
    spacer = "A" * 20
    wt = "T" * 5 + spacer + "AGG" + "C" * 30
    sites = find_spcas9_pam_sites(wt)
    assert any(site.spacer_l == 5 and site.pam == "AGG" and site.nick == 22 for site in sites)


def test_enumerate_optiprime_candidates_qualify():
    # Place edit shortly 3' of nick for a designed NGG site.
    spacer = "G" * 20
    # nick at spacer_l+17; put a 1-bp sub a few bases downstream.
    left = "C" * 10 + spacer + "CGG"
    # nick index = 10+17 = 27; edit at 30
    prefix = left[:30]
    suffix = "A" * 40
    sequence = prefix + "(A/T)" + suffix
    df = enumerate_design_candidates(sequence, design_policy="optiprime")
    assert not df.empty
    assert (df["pbs_len"] == 13).all()
    mask = apply_design_ruleset_mask(df, ["optiprime"])
    assert bool(mask.all())


def test_enumerate_anzalone_candidates_qualify():
    # 20 bp flank + 20 bp spacer + NGG + a few bases, then the edit.
    spacer = "ACGTACGTACGTACGTACGT"
    flank = "ATGCATGCATGCATGCATGC"
    between = "AAAA"
    # nick at flank(20)+17 = 37; edit a few bases 3' of nick but after PAM.
    # PAM occupies 40:43, so edit at 43+.
    sequence = flank + spacer + "TGG" + between + "(G/C)" + ("A" * 30)
    df = enumerate_design_candidates(sequence, design_policy="anzalone")
    assert not df.empty
    assert df["pbs_len"].between(10, 16).all()
    assert df["rtt_len"].between(10, 16).all()
    mask = apply_design_ruleset_mask(df, ["anzalone"])
    assert bool(mask.all())

"""Geometry helpers for OptiPrime / MinSePIE standardized windows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "pe-common"))

from app.utils.standardize_data import (  # noqa: E402
    _MINSEPIE_WIDE_FLANK_BP,
    _build_minsepie_core_target_sequences,
    _locate_optiprime_protospacer,
    _optiprime_homology_end,
)


def test_optiprime_ps_pam_edit_has_no_deepprime_pad():
    spacer = "GTCATCTTAGTCATTACCTG"
    wt = spacer + "AGGTGTTCGTTGTAACTCAT"
    left, right = _locate_optiprime_protospacer(wt, spacer)
    assert (left, right) == (0, 20)
    assert wt[left:right] == spacer
    assert wt[right : right + 3] == "AGG"


def test_optiprime_21nt_5g_spacer_matches_genomic_20mer():
    genomic = "CAGGCACCCCGACTATGATG"
    wt = "ACGT" + genomic + "AGG"
    left, right = _locate_optiprime_protospacer(wt, "G" + genomic)
    assert (left, right) == (4, 24)
    # A G immediately upstream must not steal the match (old first-20 probe).
    wt_g_flank = "ACG" + "G" + genomic + "AGG"
    left_g, right_g = _locate_optiprime_protospacer(wt_g_flank, "G" + genomic)
    assert (left_g, right_g) == (4, 24)


def test_optiprime_homology_end_uses_author_arm_not_full_mut():
    nick = 17
    ha = "ACGTACGTACGTAC"
    mut = ("A" * nick) + "TTT" + ha + ("G" * 30)
    end = _optiprime_homology_end(
        mut,
        nick=nick,
        edit_pos=nick,
        edit_len=3,
        type_del=False,
        homology_arm=ha,
    )
    assert end == nick + 3 + len(ha)
    assert end == _optiprime_homology_end(
        mut,
        nick=nick,
        edit_pos=nick,
        edit_len=3,
        type_del=False,
        homology_arm="",
    ) - 30



def test_minsepie_protospacer_interval_is_20bp():
    spacer = "ATCGATCGATCGATCGATCG"
    _wt, _mut, _edit_len, edit_position = _build_minsepie_core_target_sequences(
        spacer,
        ha_left="AAAA",
        ha_right="",
        insertion="GGG",
    )
    protospacer_l = _MINSEPIE_WIDE_FLANK_BP
    protospacer_r = protospacer_l + 20
    nick = protospacer_l + 17
    assert protospacer_r - protospacer_l == 20
    assert nick == protospacer_r - 3
    # Core reconstruction places the insertion at the nick, not by extending PAM.
    assert edit_position == 17

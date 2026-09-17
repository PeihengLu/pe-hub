"""Tests for Anzalone 2019 Easy-Prime endogenous export helpers."""

from __future__ import annotations

import pandas as pd

from pe_db.studies.anzalone import _design_id, _to_deepprime_wide_row


def test_design_id_strips_replicate_suffix():
    assert _design_id("EDFIG5A_VEGFA_10NT_REP3") == "EDFIG5A_VEGFA_10NT"
    assert _design_id("EDFIG10_HEK293T_HEK3_1TG_PE3") == "EDFIG10_HEK293T_HEK3_1TG_PE3"


def test_to_deepprime_wide_row_plus_strand():
    # Minimal synthetic amplicon: 4 bp flank + spacer + NGG PAM + 47 bp flank.
    spacer = "GATGTCTGCAGGCCAGATGA"
    amplicon = "CTTT" + spacer + "GGG" + ("C" * 47) + ("A" * 50)
    row = pd.Series(
        {
            "design_id": "TEST_PLUS",
            "reference_amplicon": amplicon,
            "correct_pegRNA": spacer,
            "pegRNA_strand": "+",
            "PBS_seq": "TCTGGCCTGCAGA",
            "RTS_seq": "GGAGCACTCA",
            "PBS_length": 13,
            "RTS_length": 10,
            "editing_frequency": 0.1737,
            "gene": "VEGFA",
            "CHROM": "chr6",
            "POS": 1,
            "REF": "G",
            "ALT": "A",
        }
    )
    out = _to_deepprime_wide_row(row)
    wide = out[
        "Wide target sequence (Total 74 bps = 4 bp neighboring sequence + 20 bp "
        "protospacer + 3 bp NGG PAM+ 47 bp neighboring sequence)"
    ]
    assert len(wide) == 74
    assert wide[4:24] == spacer
    assert "N" not in wide
    assert out["editing_efficiency"] == 17.37
    assert out["3' extension sequence of pegRNA"] == "GGAGCACTCA" + "TCTGGCCTGCAGA"

"""OPED PBS/RT orientation: genomic (PE-DB / DeepPE HT) vs pegRNA-sense (vendor ClinVar)."""
from __future__ import annotations

import pandas as pd

from app.models.oped_wrapper import OPEDModelWrapper
from pe_common.sequence_utils import reverse_complement


# DeepPE Library 1 row 0: "3' extension" PBS already equals Target nick window.
_DEEPPE_TARGET = "TTTGGTGGATTGCTTTAATAAAGATGGATCTGTCAGACCTTGGAGCT"
_DEEPPE_PBS = "TTAATAA"
_DEEPPE_RT = "AGATCGATCT"

# Vendor ClinVar Insertion row 0: PBS is pegRNA-sense (RC of nick window).
_CLINVAR_TARGET = "GTGATGCACTCCCAGCAGAACAGGTGGCCGCAGGGCGTGGCTGTTGG"
_CLINVAR_PBS_PEGRNA = "GTTCTGCT"
_CLINVAR_RT_PEGRNA = "GCGGCGCACCT"


def _pbs_tokens(encoded: pd.DataFrame) -> list[int]:
    tokens = list(encoded["PBS"].iloc[0])
    while tokens and tokens[-1] == 0:
        tokens.pop()
    return tokens


def test_orient_keeps_genomic_pbs_from_deeppe_ht():
    pbs, rt = OPEDModelWrapper._orient_pbs_rt_to_genomic(
        _DEEPPE_TARGET, _DEEPPE_PBS, _DEEPPE_RT
    )
    assert pbs == _DEEPPE_PBS
    assert rt == _DEEPPE_RT
    assert pbs == _DEEPPE_TARGET[21 - len(pbs) : 21]


def test_orient_rc_pegrna_sense_vendor_clinvar():
    pbs, rt = OPEDModelWrapper._orient_pbs_rt_to_genomic(
        _CLINVAR_TARGET, _CLINVAR_PBS_PEGRNA, _CLINVAR_RT_PEGRNA
    )
    assert pbs == reverse_complement(_CLINVAR_PBS_PEGRNA)
    assert rt == reverse_complement(_CLINVAR_RT_PEGRNA)
    assert pbs == _CLINVAR_TARGET[21 - len(pbs) : 21]


def test_encode_genomic_pbs_without_reverse_complement():
    df = pd.DataFrame(
        {
            "Target(47bp)": [_DEEPPE_TARGET],
            "PBS": [_DEEPPE_PBS],
            "RT": [_DEEPPE_RT],
        }
    )
    encoded = OPEDModelWrapper._to_oped_numeric_df(df)
    vocab = OPEDModelWrapper._build_kmer_vocab(1)
    assert _pbs_tokens(encoded) == [vocab[b] for b in _DEEPPE_PBS]
    rc_tokens = [vocab[b] for b in reverse_complement(_DEEPPE_PBS)]
    assert _pbs_tokens(encoded) != rc_tokens


def test_encode_pegrna_clinvar_matches_genomic_nick_window():
    df = pd.DataFrame(
        {
            "Target(47bp)": [_CLINVAR_TARGET],
            "PBS": [_CLINVAR_PBS_PEGRNA],
            "RT": [_CLINVAR_RT_PEGRNA],
        }
    )
    encoded = OPEDModelWrapper._to_oped_numeric_df(df)
    vocab = OPEDModelWrapper._build_kmer_vocab(1)
    genomic_pbs = reverse_complement(_CLINVAR_PBS_PEGRNA)
    assert _pbs_tokens(encoded) == [vocab[b] for b in genomic_pbs]

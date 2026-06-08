"""Tests for DeepSpCas9 target extraction and score backfill."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.deepspcas9 import (  # noqa: E402
    extract_deepspcas9_target30,
    fill_missing_spcas9_scores,
)


def _wt74() -> str:
    upstream = "ACGT"
    guide = "ATCGATCGATCGATCGATCG"
    pam = "AGG"
    downstream = "GCTA" + ("T" * 43)
    seq = upstream + guide + pam + downstream
    assert len(seq) == 74
    return seq


def test_extract_target30_matches_deepprime_window():
    wt = _wt74()
    target30 = extract_deepspcas9_target30(wt, protospacer_location_l=4)
    assert target30 == wt[:30]


def test_extract_target30_for_minsepie_layout():
    guide = "ATCGATCGATCGATCGATCG"
    wt = ("N" * 100) + guide + "AGG" + ("T" * 100)
    target30 = extract_deepspcas9_target30(wt, protospacer_location_l=100)
    assert target30 is not None
    assert len(target30) == 30
    assert target30[4:24] == guide
    assert target30[24:27] == "AGG"


def test_extract_target30_returns_none_when_out_of_bounds():
    assert extract_deepspcas9_target30("ACGT", protospacer_location_l=2) is None


def test_fill_missing_spcas9_scores_deduplicates_and_preserves_existing():
    wt = _wt74()
    df = pd.DataFrame(
        {
            "wt_sequence": [wt, wt, wt],
            "protospacer_location_l": [4, 4, 4],
            "spcas9_score": [12.5, np.nan, np.nan],
        }
    )

    def fake_scorer(targets: list[str]) -> list[float]:
        assert targets == [wt[:30]]
        return [42.0]

    filled = fill_missing_spcas9_scores(df, score_fn=fake_scorer)
    assert filled["spcas9_score"].tolist() == [12.5, 42.0, 42.0]

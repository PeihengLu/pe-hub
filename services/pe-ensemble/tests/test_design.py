"""Tests for interactive design request validation and ranking helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pe_ensemble.design.runner import _design_result_rows, _records_for_json
from pe_ensemble.design.schemas import DesignRequest


def test_records_for_json_strips_nan_and_inf():
    df = pd.DataFrame(
        {
            "spcas9_score": [float("nan"), float("inf")],
            "editing_efficiency": [0.5, float("-inf")],
            "edit_len": [1, 2],
        }
    )
    records = _records_for_json(df)
    # Must be encodable by stdlib json (requests uses this path).
    encoded = json.dumps({"records": records})
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert records[0]["spcas9_score"] is None
    assert records[0]["editing_efficiency"] == pytest.approx(0.5)
    assert records[1]["spcas9_score"] is None
    assert records[1]["editing_efficiency"] is None


def test_design_request_requires_weights_in_single_mode():
    with pytest.raises(Exception):
        DesignRequest(
            sequence="AAAA(A/G)TTTT",
            mode="single",
            model_name="deepprime",
        )


def test_design_request_ensemble_needs_two_members():
    with pytest.raises(Exception):
        DesignRequest(
            sequence="AAAA(A/G)TTTT",
            mode="ensemble",
            members=[{"model_name": "deepprime", "weights": "DeepPrime_base"}],
        )


def test_design_result_rows_sort_descending():
    std = pd.DataFrame(
        {
            "pbs_len": [13, 13],
            "rtt_len": [12, 14],
            "homology_len": [10, 10],
            "spacer": ["A" * 20, "C" * 20],
            "pam": ["AGG", "CGG"],
            "nick": [17, 17],
            "protospacer_location_l": [0, 0],
            "protospacer_location_r": [20, 20],
            "pbs_location_l": [4, 4],
            "pbs_location_r": [17, 17],
            "rtt_location_l": [17, 17],
            "rtt_location_r": [29, 31],
            "type_sub": [True, True],
            "type_ins": [False, False],
            "type_del": [False, False],
            "edit_len": [1, 1],
            "wt_sequence": ["A" * 40, "A" * 40],
            "mut_sequence": ["A" * 40, "A" * 40],
        }
    )
    rows = _design_result_rows(std, np.asarray([0.2, 0.9]))
    assert [row["rank"] for row in rows] == [1, 2]
    assert rows[0]["score"] == pytest.approx(0.9)
    assert rows[0]["spacer"] == "C" * 20

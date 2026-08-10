"""Regression tests for PRIDICT2 batch seq_id collation during prediction."""

from __future__ import annotations

import pytest
import torch

from app.models import pridict2_wrapper  # noqa: F401 — vendor import path setup
from pridict2.pridict.pridictv2.predict_outcomedistrib import _batch_seq_ids_to_list


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (("a", "b", "c"), ["a", "b", "c"]),
        (["x", "y"], ["x", "y"]),
        (torch.tensor([1, 2, 3]), [1, 2, 3]),
        ("solo", ["solo"]),
        (42, [42]),
    ],
)
def test_batch_seq_ids_to_list(value, expected):
    assert _batch_seq_ids_to_list(value) == expected

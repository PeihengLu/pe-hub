"""Ensemble prediction alignment must not silently truncate labels."""
from __future__ import annotations

import pandas as pd
import pytest

from pe_ensemble.ensemble.runner import EnsembleError, _align_member_matrix


def test_align_member_matrix_raises_on_length_mismatch():
    std_df = pd.DataFrame({"editing_efficiency": [0.1, 0.2, 0.3]})
    member_frames = [pd.DataFrame({"x": [1, 2]})]
    member_predictions = [[0.1, 0.2]]
    with pytest.raises(EnsembleError, match="Refusing to truncate"):
        _align_member_matrix(std_df, member_frames, member_predictions)


def test_align_member_matrix_keeps_all_rows_when_aligned():
    std_df = pd.DataFrame({"editing_efficiency": [0.1, 0.2, 0.3]})
    member_frames = [
        pd.DataFrame({"x": [1, 2, 3]}),
        pd.DataFrame({"x": [4, 5, 6]}),
    ]
    member_predictions = [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    y_true, matrix, alignment = _align_member_matrix(
        std_df, member_frames, member_predictions
    )
    assert list(y_true) == [0.1, 0.2, 0.3]
    assert matrix.shape == (3, 2)
    assert alignment["rows_after_alignment"] == 3
    assert "truncated" not in alignment

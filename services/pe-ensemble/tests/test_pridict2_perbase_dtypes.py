"""Regression tests for PRIDICT2 per-base nucleotide encoding dtypes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models import pridict2_wrapper  # noqa: F401 — vendor import path setup
from pridict2.pridict.pridictv2.data_preprocess import PESeqProcessor


def test_process_perbase_df_encodes_b_columns_as_int64():
    """Pandas 3 StringDtype leaves object arrays after replace unless cast."""
    proc = PESeqProcessor()
    df = pd.DataFrame(
        {
            "seq_id": ["seq_0", "seq_1"],
            "wide_initial_target_align": ["ACGT", "AC-G"],
        }
    )
    out, _num_cols = proc.process_perbase_df(df, "seq_id", "wide_initial_target_align")
    b_cols = [c for c in out.columns if c.startswith("B")]
    assert b_cols
    assert out[b_cols].values.dtype == np.int64

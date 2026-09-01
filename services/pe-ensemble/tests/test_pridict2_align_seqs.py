"""Regression tests for PRIDICT2 sequence alignment preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from app.models import pridict2_wrapper  # noqa: F401 — vendor import path setup
from pridict2.pridict.pridictv2.data_preprocess import PESeqProcessor


def _minimal_align_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seq_id": [f"seq_{i}" for i in range(n)],
            "wide_initial_target": ["ACGTACGT"] * n,
            "wide_mutated_target": ["ACGTACGT"] * n,
            "deepeditposition": [3] * n,
            "deepeditposition_lst": ["[3]"] * n,
            "Correction_Type": ["Replacement"] * n,
            "Correction_Length": [1] * n,
            "protospacerlocation_only_initial": ["[1, 8]"] * n,
            "PBSlocation": ["[1, 4]"] * n,
            "RT_initial_location": ["[5, 8]"] * n,
            "RT_mutated_location": ["[5, 8]"] * n,
        }
    )


def test_align_seqs_keeps_unique_seq_id_after_groupby():
    """Unique seq_id rows survive align_seqs (merge key preserved)."""
    proc = PESeqProcessor()
    df = _minimal_align_df()

    aligned = proc.align_seqs(df, "wide_initial_target", "wide_mutated_target")

    assert "seq_id" in aligned.columns
    assert aligned["seq_id"].tolist() == df["seq_id"].tolist()
    assert len(aligned) == len(df)


def test_align_seqs_survives_pandas_include_groups_default():
    """Regression: reset_index() with include_groups=True duplicated seq_id."""
    proc = PESeqProcessor()
    df = _minimal_align_df(n=5)
    # Mixed edit types like library1.
    df.loc[1, "Correction_Type"] = "Insertion"
    df.loc[1, "Correction_Length"] = 2
    df.loc[1, "wide_mutated_target"] = "ACGTTACGT"
    df.loc[2, "Correction_Type"] = "Deletion"
    df.loc[2, "Correction_Length"] = 1
    df.loc[2, "wide_initial_target"] = "ACGTTACGT"
    aligned = proc.align_seqs(df, "wide_initial_target", "wide_mutated_target")
    assert aligned["seq_id"].is_unique
    assert {"wide_initial_target_align", "wide_mutated_target_align"}.issubset(aligned.columns)

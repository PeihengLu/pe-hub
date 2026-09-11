"""DeepPE drops unmeasured efficiency rows during study standardization."""
from __future__ import annotations

import pandas as pd

from pe_db.studies.deeppe import (
    _average_deeppe_replicates,
    _drop_unlabeled_efficiency_rows,
    _prepare_deeppe_export_df,
)


def test_drop_unlabeled_efficiency_keeps_genuine_zero():
    df = pd.DataFrame({"editing_efficiency": [0.0, 0.4, None, ""]})
    kept = _drop_unlabeled_efficiency_rows(df, "editing_efficiency")
    assert kept["editing_efficiency"].tolist() == [0.0, 0.4]


def test_all_nan_replicate_mean_is_dropped_as_unmeasured():
    df = pd.DataFrame(
        {
            "Endo-BR1-TR1": [1.0, None],
            "Endo-BR1-TR2": [3.0, None],
            "Endo-BR2-TR1": [None, None],
            "Endo-BR2-TR2": [None, None],
        }
    )
    df["editing_efficiency"] = _average_deeppe_replicates(
        df, ["Endo-BR1-TR1", "Endo-BR1-TR2", "Endo-BR2-TR1", "Endo-BR2-TR2"]
    )
    kept = _drop_unlabeled_efficiency_rows(df, "editing_efficiency")
    assert len(kept) == 1
    assert kept["editing_efficiency"].tolist() == [2.0]


def _toy_deeppe_rows() -> pd.DataFrame:
    """Two 47 bp reporter rows; the second has no measured efficiency."""
    spacer = "G" * 20
    wt = "ACGT" + spacer + "GGG" + "C" * 20
    assert len(wt) == 47
    pbs_len, rt_len = 13, 15
    pbs_rt = pbs_len + rt_len
    neighbor = 4
    rt_mut = "A" + wt[neighbor + pbs_len + 1 : neighbor + pbs_rt]
    mut = (
        "X" * neighbor
        + wt[neighbor : neighbor + pbs_len]
        + rt_mut
        + "X" * (len(wt) - neighbor - pbs_rt)
    )
    assert len(mut) == 47
    return pd.DataFrame(
        {
            "wt_sequence": [wt, wt],
            "mut_sequence": [mut, mut],
            "pbslen": [pbs_len, pbs_len],
            "rtlen": [rt_len, rt_len],
            "editing_efficiency": [0.12, None],
        }
    )


def test_prepare_deeppe_export_df_drops_unlabeled_efficiency_rows():
    prepared = _prepare_deeppe_export_df(_toy_deeppe_rows())
    assert len(prepared) == 1
    assert prepared["measured_pe_efficiency"].tolist() == [0.12]
    assert prepared["measured_pe_efficiency"].notna().all()

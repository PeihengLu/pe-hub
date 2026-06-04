"""Tests for standardized -> model-format conversions.

Guards the canonical ``edit_len`` column handling (a prior bug read the
non-existent ``edit_length`` column and silently produced edit length 0) and the
output schemas required by each vendor model.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "pe-common"))

from app.utils.convert_data import (  # noqa: E402
    is_standardized_dataframe,
    standardized_to_deepprime_dataframe,
    standardized_to_oped_dataframe,
    standardized_to_pridict_dataframe,
)
from app.utils.standardize_data import (  # noqa: E402
    _build_standardized_output_df,
    _coerce_original_fold,
)

DEEPPRIME_REQUIRED = {
    "WT74_On", "Edited74_On", "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos",
    "Edit_len", "RHA_len", "type_sub", "type_ins", "type_del", "DeepSpCas9_score",
}
PRIDICT_REQUIRED = {
    "seq_id", "wide_initial_target", "wide_mutated_target", "deepeditposition",
    "Correction_Type", "Correction_Length", "protospacerlocation_only_initial",
    "PBSlocation", "RT_initial_location", "RT_mutated_location",
}
OPED_REQUIRED = {"Target(47bp)", "PBS", "RT"}


def _standardized_df(edit_length_col: str = "edit_len") -> pd.DataFrame:
    base = {
        "wt_sequence": ["ACGT" * 30, "TGCA" * 30],
        "mut_sequence": ["ACGT" * 30, "TGCA" * 30],
        edit_length_col: [1, 3],
        "type_sub": [True, False],
        "type_ins": [False, True],
        "type_del": [False, False],
        "protospacer_location_l": [10, 12],
        "protospacer_location_r": [30, 32],
        "pbs_location_l": [20, 22],
        "pbs_location_r": [33, 35],
        "rtt_location_l": [33, 35],
        "rtt_location_r": [50, 52],
        "lha_location_r": [40, 42],
        "rha_location_l": [41, 43],
        "rha_location_r": [48, 50],
        "editing_efficiency": [0.3, 0.7],
        "spcas9_score": [0.5, 0.6],
    }
    return pd.DataFrame(base)


def test_is_standardized_uses_edit_len():
    df = _standardized_df("edit_len")
    assert is_standardized_dataframe(df) is True


def test_deepprime_edit_len_is_not_zeroed():
    df = _standardized_df("edit_len")
    out = standardized_to_deepprime_dataframe(df)
    assert DEEPPRIME_REQUIRED.issubset(out.columns)
    assert out["Edit_len"].tolist() == [1, 3]
    assert out["WT74_On"].str.len().eq(74).all()
    assert out["Edited74_On"].str.len().eq(74).all()


def test_pridict_correction_length_matches_edit_len():
    df = _standardized_df("edit_len")
    out = standardized_to_pridict_dataframe(df)
    assert PRIDICT_REQUIRED.issubset(out.columns)
    assert out["Correction_Length"].tolist() == [1, 3]


def test_oped_schema_and_target_length():
    df = _standardized_df("edit_len")
    out = standardized_to_oped_dataframe(df)
    assert OPED_REQUIRED.issubset(out.columns)
    assert out["Target(47bp)"].str.len().eq(47).all()


def test_legacy_edit_length_column_still_supported():
    df = _standardized_df("edit_length")
    out = standardized_to_deepprime_dataframe(df)
    assert out["Edit_len"].tolist() == [1, 3]


def test_original_fold_unknown_is_nan():
    assert _coerce_original_fold(None, length=3).isna().all()
    partial = pd.Series([0.0, None, 2.0])
    coerced = _coerce_original_fold(partial, length=3)
    assert coerced.iloc[0] == 0.0
    assert pd.isna(coerced.iloc[1])
    assert coerced.iloc[2] == 2.0


def test_build_standardized_output_df_omits_fold_when_unknown():
    base = _standardized_df("edit_len")
    out = _build_standardized_output_df(
        group_id=base.index,
        type_sub=base["type_sub"],
        type_ins=base["type_ins"],
        type_del=base["type_del"],
        edit_len=base["edit_len"],
        wt_sequence=base["wt_sequence"],
        mut_sequence=base["mut_sequence"],
        protospacer_location_l=base["protospacer_location_l"],
        protospacer_location_r=base["protospacer_location_r"],
        pbs_location_l=base["pbs_location_l"],
        pbs_location_r=base["pbs_location_r"],
        rtt_location_l=base["rtt_location_l"],
        rtt_location_r=base["rtt_location_r"],
        lha_location_l=base["protospacer_location_l"],
        lha_location_r=base["lha_location_r"],
        rha_location_l=base["rha_location_l"],
        rha_location_r=base["rha_location_r"],
        spcas9_score=base["spcas9_score"],
        editing_efficiency=base["editing_efficiency"],
    )
    assert out["original_fold"].isna().all()

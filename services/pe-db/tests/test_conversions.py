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
    PRIDICT2_NORMALIZER_COLUMNS,
    _compute_deepprime_thermo_features,
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
    "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4",
}
PRIDICT_REQUIRED = {
    "seq_id", "wide_initial_target", "wide_mutated_target", "deepeditposition",
    "Correction_Type", "Correction_Length", "protospacerlocation_only_initial",
    "PBSlocation", "RT_initial_location", "RT_mutated_location",
}
OPED_REQUIRED = {"Target(47bp)", "PBS", "RT"}

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIDICT2_EXPORT = REPO_ROOT / "datasets" / "exported" / "pridict2" / "library-diverse" / "hek-pe2.csv"
PRIDICT2_PARQUET = REPO_ROOT / "datasets" / "standardized" / "pridict2" / "library_diverse" / "hek-pe2.parquet"

PRIDICT2_AUTHOR_FEATURE_MAP = {
    "RToverhangmatches": "RToverhangmatches",
    "PBSlength": "PBSlength",
    "RTlength": "RTTlength",
    "RToverhanglength": "RTToverhanglength",
    "MFE_protospacer": "MFE_protospacer",
    "MFE_protospacer_scaffold": "MFE_protospacer_scaffold",
    "MFE_extension": "MFE_extension",
    "MFE_extension_scaffold": "MFE_extension_scaffold",
    "MFE_protospacer_extension_scaffold": "MFE_protospacer_extension_scaffold",
    "MFE_rt": "MFE_rt",
    "MFE_pbs": "MFE_pbs",
    "protospacermt": "protospacermt",
    "extensionmt": "extensionmt",
    "RTmt": "RTmt",
    "RToverhangmt": "RToverhangmt",
    "PBSmt": "PBSmt",
    "original_base_mt": "original_base_mt",
    "edited_base_mt": "edited_base_mt",
    "nGCcnt1": "PBS_GC_count",
    "nGCcnt2": "RT_GC_count",
    "nGCcnt3": "Extension_GC_count",
    "fGCcont1": "PBS_GC_content",
    "fGCcont2": "RT_GC_content",
    "fGCcont3": "Extension_GC_content",
}


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


def _parse_bounds(value: str) -> tuple[int, int]:
    left = int(value.split("[")[1].split(",")[0])
    right = int(value.split(",")[1].strip(" ]"))
    return left, right


def _export_row_to_standardized(row: pd.Series) -> pd.DataFrame:
    prot_l, prot_r = _parse_bounds(row["protospacerlocation_only_initial"])
    pbs_l, pbs_r = _parse_bounds(row["PBSlocation"])
    rtt_wt_l, rtt_wt_r = _parse_bounds(row["RT_initial_location"])
    _, rtt_mut_r = _parse_bounds(row["RT_mutated_location"])
    return pd.DataFrame(
        {
            "wt_sequence": [str(row["wide_initial_target"]).upper()],
            "mut_sequence": [str(row["wide_mutated_target"]).upper()],
            "edit_len": [int(row["Correction_Length"])],
            "type_sub": [row["Correction_Type"] == "Replacement"],
            "type_ins": [row["Correction_Type"] == "Insertion"],
            "type_del": [row["Correction_Type"] == "Deletion"],
            "protospacer_location_l": [prot_l],
            "protospacer_location_r": [prot_r],
            "pbs_location_l": [pbs_l],
            "pbs_location_r": [pbs_r],
            "rtt_location_l": [rtt_wt_l],
            "rtt_location_r": [rtt_mut_r],
            "lha_location_r": [int(row["deepeditposition"])],
            "rha_location_l": [rtt_wt_r - int(row["RTToverhanglength"])],
            "rha_location_r": [rtt_wt_r],
            "spcas9_score": [float(row["deepcas9"])],
        }
    )


def _standardized_from_parquet_row(row: pd.Series) -> pd.DataFrame:
    keep = [c for c in _standardized_df().columns if c in row.index]
    return pd.DataFrame([row[keep].to_dict()])


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
    assert (out["Tm1"] != 0).any()
    assert out["MFE3"].notna().all()
    assert out["MFE4"].notna().all()


def test_pridict_correction_length_matches_edit_len():
    df = _standardized_df("edit_len")
    out = standardized_to_pridict_dataframe(df)
    assert PRIDICT_REQUIRED.issubset(out.columns)
    assert out["Correction_Length"].tolist() == [1, 3]
    assert out["Correction_Length"].dtype in (int, "int64", "int32")


def test_pridict_includes_vendor_normalizer_columns():
    df = _standardized_df("edit_len")
    out = standardized_to_pridict_dataframe(df)
    for colname in PRIDICT2_NORMALIZER_COLUMNS:
        assert colname in out.columns
    assert "Correction_Length_effective" not in out.columns


def test_pridict_conversion_reports_progress():
    df = _standardized_df("edit_len")
    messages: list[str] = []
    standardized_to_pridict_dataframe(df, progress_callback=messages.append)
    assert messages
    assert any("thermodynamic" in message.lower() for message in messages)
    assert messages[-1].endswith("(100%)")


@pytest.mark.skipif(not PRIDICT2_EXPORT.is_file(), reason="PRIDICT2 export fixture unavailable")
def test_pridict_features_match_author_export_from_minimal_standardized():
    export = pd.read_csv(PRIDICT2_EXPORT, nrows=100, low_memory=False)
    for _, row in export.iterrows():
        std = _export_row_to_standardized(row)
        out = standardized_to_pridict_dataframe(std).iloc[0]
        for out_col, author_col in PRIDICT2_AUTHOR_FEATURE_MAP.items():
            assert out[out_col] == pytest.approx(float(row[author_col]), rel=0.0, abs=0.05), out_col


@pytest.mark.skipif(
    not PRIDICT2_PARQUET.is_file(),
    reason="standardized PRIDICT2 parquet unavailable",
)
def test_pridict_features_match_author_export_from_standardized_parquet():
    parquet = pd.read_parquet(PRIDICT2_PARQUET).head(100)
    export = pd.read_csv(PRIDICT2_EXPORT, nrows=100, low_memory=False)
    for i in range(len(parquet)):
        std = _standardized_from_parquet_row(parquet.iloc[i])
        out = standardized_to_pridict_dataframe(std).iloc[0]
        row = export.iloc[i]
        assert out["RToverhangmatches"] == pytest.approx(float(row["RToverhangmatches"]))
        assert out["MFE_extension"] == pytest.approx(float(row["MFE_extension"]), abs=0.05)


@pytest.mark.skipif(not PRIDICT2_EXPORT.is_file(), reason="PRIDICT2 export fixture unavailable")
def test_deepprime_thermo_features_computed_from_standardized_schema():
    export = pd.read_csv(PRIDICT2_EXPORT, nrows=50, low_memory=False)
    for _, row in export.iterrows():
        std = _export_row_to_standardized(row).iloc[0]
        out = standardized_to_deepprime_dataframe(pd.DataFrame([std])).iloc[0]
        wt = std["wt_sequence"]
        mut = std["mut_sequence"]
        pbs = mut[std["pbs_location_l"]:std["pbs_location_r"]]
        rt = mut[std["rtt_location_l"]:std["rtt_location_r"]]
        ref = _compute_deepprime_thermo_features(
            wt,
            pbs,
            rt,
            protospacer_l=int(std["protospacer_location_l"]),
            protospacer_r=int(std["protospacer_location_r"]),
            edit_len=int(std["edit_len"]),
            type_sub=bool(std["type_sub"]),
            type_ins=bool(std["type_ins"]),
            type_del=bool(std["type_del"]),
        )
        for col in ref:
            assert out[col] == pytest.approx(ref[col], abs=0.05), col


def test_pridict_rt_initial_location_accounts_for_indels():
    """rtt_location_r is the mutated RT end; WT RT end must be adjusted on indels."""
    insertion = pd.DataFrame(
        {
            "wt_sequence": ["A" * 60],
            "mut_sequence": ["A" * 72],
            "edit_len": [12],
            "type_sub": [False],
            "type_ins": [True],
            "type_del": [False],
            "protospacer_location_l": [10],
            "protospacer_location_r": [30],
            "pbs_location_l": [20],
            "pbs_location_r": [26],
            "rtt_location_l": [26],
            "rtt_location_r": [50],
            "lha_location_r": [38],
            "rha_location_l": [37],
            "rha_location_r": [38],
            "spcas9_score": [0.5],
        }
    )
    deletion = insertion.copy()
    deletion["mut_sequence"] = ["A" * 48]
    deletion["edit_len"] = [12]
    deletion["type_ins"] = [False]
    deletion["type_del"] = [True]
    deletion["rtt_location_r"] = [38]
    deletion["lha_location_r"] = [26]
    deletion["rha_location_l"] = [25]
    deletion["rha_location_r"] = [26]

    ins_out = standardized_to_pridict_dataframe(insertion)
    del_out = standardized_to_pridict_dataframe(deletion)

    assert ins_out["RT_initial_location"].iloc[0] == "[26, 38]"
    assert ins_out["RT_mutated_location"].iloc[0] == "[26, 50]"
    assert del_out["RT_initial_location"].iloc[0] == "[26, 50]"
    assert del_out["RT_mutated_location"].iloc[0] == "[26, 38]"


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

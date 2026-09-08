"""200 bp endogenous PE-core window helpers and converter crops."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages" / "pe-common"))

from pe_common.sequence_utils import remove_padding, unpadded_coordinate  # noqa: E402
from pe_db.pipeline.endo import (  # noqa: E402
    ENDO_CONTEXT_BP,
    ENDO_SPACER_LEN,
    ENDO_SPACER_OFFSET,
    expand_aligned_to_endo_context,
    expand_endogenous_frame,
    genomic_window_interval,
    slice_reference_window_to_endo_context,
)
from pe_db.utils.convert_data import (  # noqa: E402
    standardized_to_deepprime_dataframe,
    standardized_to_oped_dataframe,
    standardized_to_pridict_dataframe,
)


def _synthetic_window(spacer: str = "ACGTACGTACGTACGTACGT") -> str:
    assert len(spacer) == ENDO_SPACER_LEN
    return ("A" * ENDO_SPACER_OFFSET) + spacer + ("C" * (ENDO_CONTEXT_BP - ENDO_SPACER_OFFSET - ENDO_SPACER_LEN))


def test_genomic_window_interval_is_200bp_on_both_strands():
    start, end = genomic_window_interval(1000)
    assert end - start == ENDO_CONTEXT_BP
    assert start == 1000 - ENDO_SPACER_OFFSET
    assert end == 1000 + (ENDO_CONTEXT_BP - ENDO_SPACER_OFFSET)


def test_slice_minsepie_220_to_shared_200():
    spacer = "G" * 20
    old = ("T" * 100) + spacer + ("A" * 100)
    sliced, offset = slice_reference_window_to_endo_context(old, 100)
    assert offset == ENDO_SPACER_OFFSET
    assert len(sliced) == ENDO_CONTEXT_BP
    assert sliced[ENDO_SPACER_OFFSET : ENDO_SPACER_OFFSET + 20] == spacer
    assert sliced == old[10:210]


def test_expand_deeppe_like_47bp_row_to_200():
    window = _synthetic_window()
    spacer = window[ENDO_SPACER_OFFSET : ENDO_SPACER_OFFSET + 20]
    author = window[ENDO_SPACER_OFFSET - 4 : ENDO_SPACER_OFFSET - 4 + 47]
    assert author[4:24] == spacer
    wt = author[:21] + ("N" * 3) + author[21:]
    mut = author[:21] + "TTT" + author[21:]
    result = expand_aligned_to_endo_context(wt, mut, 4, 24, window)
    assert result is not None
    new_wt, new_mut, shift = result
    assert shift == ENDO_SPACER_OFFSET - 4
    assert unpadded_coordinate(new_wt, 4 + shift) == ENDO_SPACER_OFFSET
    assert remove_padding(new_wt)[ENDO_SPACER_OFFSET : ENDO_SPACER_OFFSET + 20] == spacer
    assert len(remove_padding(new_wt)) == ENDO_CONTEXT_BP
    assert "NNN" in new_wt
    assert "TTT" in new_mut


def test_expand_endogenous_frame_shifts_geometry():
    window = _synthetic_window()
    author = window[ENDO_SPACER_OFFSET - 10 : ENDO_SPACER_OFFSET - 10 + 99]
    df = pd.DataFrame(
        {
            "wt_sequence": [author],
            "mut_sequence": [author],
            "protospacer_location_l": [10],
            "protospacer_location_r": [30],
            "pbs_location_l": [15],
            "pbs_location_r": [27],
            "rtt_location_l": [27],
            "rtt_location_r": [50],
            "lha_location_l": [27],
            "lha_location_r": [35],
            "rha_location_l": [40],
            "rha_location_r": [50],
        }
    )
    out = expand_endogenous_frame(df, pd.Series([window]))
    assert out["protospacer_location_l"].iloc[0] == ENDO_SPACER_OFFSET
    assert out["pbs_location_l"].iloc[0] == 15 + (ENDO_SPACER_OFFSET - 10)
    assert len(remove_padding(out["wt_sequence"].iloc[0])) == ENDO_CONTEXT_BP


def test_converters_crop_200bp_endogenous_row():
    window = _synthetic_window()
    spacer = window[ENDO_SPACER_OFFSET : ENDO_SPACER_OFFSET + 20]
    wt = window
    mut = window[: ENDO_SPACER_OFFSET + 17] + "G" + window[ENDO_SPACER_OFFSET + 18 :]
    df = pd.DataFrame(
        {
            "wt_sequence": [wt],
            "mut_sequence": [mut],
            "edit_len": [1],
            "type_sub": [True],
            "type_ins": [False],
            "type_del": [False],
            "protospacer_location_l": [ENDO_SPACER_OFFSET],
            "protospacer_location_r": [ENDO_SPACER_OFFSET + 20],
            "pbs_location_l": [ENDO_SPACER_OFFSET + 4],
            "pbs_location_r": [ENDO_SPACER_OFFSET + 17],
            "rtt_location_l": [ENDO_SPACER_OFFSET + 17],
            "rtt_location_r": [ENDO_SPACER_OFFSET + 40],
            "lha_location_l": [ENDO_SPACER_OFFSET + 17],
            "lha_location_r": [ENDO_SPACER_OFFSET + 17],
            "rha_location_l": [ENDO_SPACER_OFFSET + 18],
            "rha_location_r": [ENDO_SPACER_OFFSET + 40],
            "editing_efficiency": [0.2],
            "spcas9_score": [0.5],
        }
    )
    deepprime = standardized_to_deepprime_dataframe(df)
    assert deepprime["WT74_On"].str.len().eq(74).all()
    assert deepprime["WT74_On"].iloc[0][4:24] == spacer
    oped = standardized_to_oped_dataframe(df)
    assert oped["Target(47bp)"].str.len().eq(47).all()
    assert oped["Target(47bp)"].iloc[0][4:24] == spacer
    pridict = standardized_to_pridict_dataframe(df)
    left = int(pridict["protospacerlocation_only_initial"].iloc[0].split("[")[1].split(",")[0])
    assert left == 10
    assert unpadded_coordinate(pridict["wide_initial_target"].iloc[0], left) == 10
    assert pridict["wide_initial_target"].iloc[0][10:30] == spacer

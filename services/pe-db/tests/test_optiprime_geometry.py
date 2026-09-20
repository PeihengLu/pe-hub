"""OptiPrime's target endpoint is part of the model's learned input schema."""
import pandas as pd
import pytest
from pe_db.formats.optiprime import standardized_to_optiprime_dataframe


@pytest.mark.parametrize("edit", ["sub", "ins", "del"])
@pytest.mark.parametrize("spacer_len", [19, 20, 21])
def test_targets_match_vendor_design_geometry(edit, spacer_len):
    wt = "ACGTGACGTACGTACGTACGTACGTAGGACCTAGCATCGATCGTAGCTAGCATCGATCG"
    mut = wt[:30] + {"sub": "A", "ins": "AAA", "del": ""}[edit] + wt[30 + (edit != "ins"):]
    delta = len(mut) - len(wt)
    wt_aligned = wt[:30] + ("N" * max(delta, 0)) + wt[30:]
    mut_aligned = mut[:30] + ("N" * max(-delta, 0)) + mut[30:]
    rtt_len = 20
    aligned_end = 21 + rtt_len + max(-delta, 0)
    frame = pd.DataFrame([dict(
        wt_sequence=wt_aligned, mut_sequence=mut_aligned,
        protospacer_location_l=24-spacer_len, protospacer_location_r=24,
        pbs_location_l=8, pbs_location_r=21,
        rtt_location_l=21, rtt_location_r=aligned_end,
        editing_efficiency=0.2,
    )])
    row = standardized_to_optiprime_dataframe(frame, study="pridict2").iloc[0]
    # Exact slices used by vendor DESIGN_PE.make_pegrna_dataset.
    assert row.full_edited == mut[:25 + rtt_len]
    assert row.full_unedited == wt[:25 + rtt_len - delta]
    assert len(row.rtt) == rtt_len
    assert row.full_unedited[4:24] == wt[4:24]


def test_extra_downstream_context_cannot_change_model_inputs():
    wt = "ACGTGACGTACGTACGTACGTACGTAGGACCTAGCATCGATCGTAGCTAGCATCGATCG"
    mut = wt[:30] + "A" + wt[31:]
    base = dict(protospacer_location_l=4, protospacer_location_r=24,
                pbs_location_l=8, pbs_location_r=21,
                rtt_location_l=21, rtt_location_r=39, editing_efficiency=0.2)
    frame = pd.DataFrame([dict(base, wt_sequence=wt, mut_sequence=mut),
                          dict(base, wt_sequence=wt[:43], mut_sequence=mut[:43])])
    out = standardized_to_optiprime_dataframe(frame)
    pd.testing.assert_series_equal(out.iloc[0], out.iloc[1], check_names=False)


@pytest.mark.parametrize("study, expected", [("deeppe", 0.005), ("deepprime", 0.005),
                                            ("pridict2", 0.005), ("optiprime", 0.5),
                                            (None, 0.5)])
def test_labels_use_fraction_units_even_for_low_percentage_subset(study, expected):
    frame = pd.DataFrame([dict(wt_sequence="ACGT" * 15, mut_sequence="ACGT" * 15,
                              protospacer_location_l=4, protospacer_location_r=24,
                              pbs_location_l=8, pbs_location_r=21,
                              rtt_location_l=21, rtt_location_r=39,
                              editing_efficiency=0.5)])
    row = standardized_to_optiprime_dataframe(frame, study=study).iloc[0]
    assert row.Efficiency == expected
    assert row.edited_frac == expected


def test_insertion_inside_spacer_does_not_shift_genomic_frame():
    wt = "ACGTGACGTACGTACGTACGTACGTAGGACCTAGCATCGATCGTAGCTAGCATCGATCG"
    mut = wt[:21] + "AAA" + wt[21:]
    frame = pd.DataFrame([dict(wt_sequence=wt[:21] + "NNN" + wt[21:],
                              mut_sequence=mut, protospacer_location_l=4,
                              protospacer_location_r=27, pbs_location_l=8,
                              pbs_location_r=21, rtt_location_l=21,
                              rtt_location_r=41, editing_efficiency=0.2)])
    row = standardized_to_optiprime_dataframe(frame).iloc[0]
    assert row.full_unedited == wt[:42]
    assert row.full_edited == mut[:45]
    assert row.proto30 == wt[:30]

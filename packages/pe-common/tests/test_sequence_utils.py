"""Tests for lossless WT/Mut alignment and pad coordinate mapping."""

from pe_common.sequence_utils import (
    align_wt_mut_sequences,
    normalize_target_dna,
    remove_padding,
    shift_coords_after_indel_pad,
    shift_index_after_indel_pad,
    unpadded_coordinate,
)


def test_insertion_grows_instead_of_truncating():
    wt, mut = align_wt_mut_sequences("ATCG", "ATGCG", edit_position=2, edit_length=1, edit_type=1)
    assert wt == "ATNCG"
    assert mut == "ATGCG"
    assert len(wt) == len(mut) == 5


def test_insertion_keeps_author_3prime_when_windows_were_equal_length():
    # 8 bp author windows with a 2 bp insertion squeezed into Mut.
    wt = "AAAACTTT"
    mut = "AAAAGGCT"
    aligned_wt, aligned_mut = align_wt_mut_sequences(
        wt, mut, edit_position=4, edit_length=2, edit_type=1
    )
    assert aligned_wt == "AAAANNCTTT"
    assert aligned_mut == "AAAAGGCTTT"
    assert aligned_wt.endswith("CTTT")
    assert "N" not in aligned_mut


def test_deletion_pads_mut_without_clipping_wt():
    wt, mut = align_wt_mut_sequences("ATGCG", "ATCG", edit_position=2, edit_length=1, edit_type=2)
    assert wt == "ATGCG"
    assert mut == "ATNCG"


def test_substitution_leaves_sequences_unchanged():
    wt, mut = align_wt_mut_sequences("ATGC", "ATCC", edit_position=2, edit_length=1, edit_type=0)
    assert wt == "ATGC"
    assert mut == "ATCC"


def test_shift_index_moves_only_positions_3prime_of_pad():
    assert shift_index_after_indel_pad(10, 12, 3, True) == 10
    assert shift_index_after_indel_pad(12, 12, 3, True) == 12
    assert shift_index_after_indel_pad(13, 12, 3, True) == 16
    assert shift_index_after_indel_pad(13, 12, 3, False) == 13


def test_shift_coords_vectorized():
    shifted = shift_coords_after_indel_pad(
        coords=[10, 20, 30],
        edit_position=[15, 15, 15],
        edit_length=[2, 2, 2],
        strand_padded=[True, True, False],
    )
    assert shifted.tolist() == [10, 22, 30]


def test_unpadded_coordinate_skips_n_pads():
    assert unpadded_coordinate("ATNNCG", 0) == 0
    assert unpadded_coordinate("ATNNCG", 2) == 2
    assert unpadded_coordinate("ATNNCG", 4) == 2
    assert unpadded_coordinate("ATNNCG", 6) == 4
    assert remove_padding("ATNNCG") == "ATCG"


def test_normalize_target_dna_keeps_n():
    assert normalize_target_dna("atnug") == "ATNTG"

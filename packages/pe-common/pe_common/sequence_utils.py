"""Sequence manipulation utilities"""
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd

_PAD_BASES = frozenset("NX-")
_DNA_WITH_PAD = frozenset("ACGTN")


def align_wt_mut_sequences(
    wt_sequence: str, 
    mut_sequence: str, 
    edit_position: int, 
    edit_length: int, 
    edit_type: int
) -> Tuple[str, str]:
    """
    Align WT and Mut by inserting ``N`` pads at the edit, then grow to equal length.

    Insertions pad WT; deletions pad Mut. The shorter strand is then extended
    with the longer partner's overhang so 3' sequence is not discarded. Vendor
    windows (74 / 47 / 99 bp) are a conversion concern, not a standardize trim.
    
    Args:
        wt_sequence: Wild type sequence
        mut_sequence: Mutated sequence
        edit_position: Position of the edit (0-indexed)
        edit_length: Length of the edit
        edit_type: Type of edit (0=substitution, 1=insertion, 2=deletion)
        
    Returns:
        Tuple of (aligned_wt_sequence, aligned_mut_sequence)
    """
    wt_sequence = str(wt_sequence)
    mut_sequence = str(mut_sequence)
    edit_position = int(edit_position)
    edit_length = max(0, int(edit_length))

    if edit_type == 1 and edit_length:  # insertion
        edit_position = min(max(edit_position, 0), len(wt_sequence))
        wt_sequence = (
            wt_sequence[:edit_position] + ("N" * edit_length) + wt_sequence[edit_position:]
        )
    elif edit_type == 2 and edit_length:  # deletion
        edit_position = min(max(edit_position, 0), len(mut_sequence))
        mut_sequence = (
            mut_sequence[:edit_position] + ("N" * edit_length) + mut_sequence[edit_position:]
        )

    if len(wt_sequence) < len(mut_sequence):
        wt_sequence = wt_sequence + mut_sequence[len(wt_sequence):]
    elif len(mut_sequence) < len(wt_sequence):
        mut_sequence = mut_sequence + wt_sequence[len(mut_sequence):]
    return wt_sequence, mut_sequence


def shift_index_after_indel_pad(
    index: int,
    edit_position: int,
    edit_length: int,
    strand_padded: bool,
) -> int:
    """Shift one coordinate 3' of an indel pad onto the grown aligned sequence."""
    if not strand_padded or int(edit_length) <= 0:
        return int(index)
    if int(index) > int(edit_position):
        return int(index) + int(edit_length)
    return int(index)


def shift_coords_after_indel_pad(
    coords: Union[int, np.ndarray, pd.Series],
    edit_position: Union[int, np.ndarray, pd.Series],
    edit_length: Union[int, np.ndarray, pd.Series],
    strand_padded: Union[bool, np.ndarray, pd.Series],
) -> pd.Series:
    """Shift coordinates 3' of an indel pad (vectorized)."""
    edit_position = pd.Series(edit_position)
    index = edit_position.index
    edit_length = pd.Series(edit_length, index=index)
    strand_padded = pd.Series(strand_padded, index=index).astype(bool)
    if np.isscalar(coords) or isinstance(coords, (int, np.integer)):
        coords = pd.Series(int(coords), index=index)
    else:
        coords = pd.Series(coords, index=index)
    shift = (
        strand_padded & (coords.astype(int) > edit_position.astype(int))
    ).astype(int) * edit_length.astype(int).clip(lower=0)
    return (coords.astype(int) + shift).astype(int)


def unpadded_coordinate(sequence: str, index: int) -> int:
    """Map a padded-sequence coordinate to the same locus after pads are dropped."""
    sequence = str(sequence)
    index = max(0, min(int(index), len(sequence)))
    return sum(1 for base in sequence[:index] if base not in _PAD_BASES)


def normalize_target_dna(sequence: str) -> str:
    """Uppercase DNA (U→T) keeping ``N`` alignment pads; other non-ACGT become ``N``."""
    sequence = str(sequence).upper().replace("U", "T")
    return "".join(base if base in _DNA_WITH_PAD else "N" for base in sequence)


def remove_padding(sequence: str) -> str:
    """
    Remove different types of padding characters from the sequence
    
    Args:
        sequence: DNA/RNA sequence with potential padding characters
        
    Returns:
        Sequence with padding characters removed
    """
    sequence = sequence.strip()
    if 'N' in sequence:
        sequence = sequence.replace('N', '')
    if 'X' in sequence:
        sequence = sequence.replace('X', '')
    if '-' in sequence:
        sequence = sequence.replace('-', '')
    return sequence


def onehot_encode(sequence: str) -> np.ndarray:
    """
    One-hot encode a DNA/RNA sequence
    
    Args:
        sequence: DNA/RNA sequence string of length n
        
    Returns:
        onehot encoded nx4 numpy array
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
               'U': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'u': 3}
    onehot = np.zeros((len(sequence), 4), dtype=np.int8)
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            onehot[i, mapping[nucleotide]] = 1
    return onehot

def reverse_complement(
    sequence: str,
    mode: Literal[
        "auto", "dna_to_dna", "rna_to_rna", "dna_to_rna", "rna_to_dna"
    ] = "auto",
) -> str:
    """
    Reverse complement a DNA/RNA sequence
    
    Args:
        sequence: DNA/RNA sequence string of length n
        mode:
            - "auto": infer from input (`U` without `T` => RNA-like, else DNA-like)
            - "dna_to_dna": DNA reverse complement with DNA output alphabet
            - "rna_to_rna": RNA reverse complement with RNA output alphabet
            - "dna_to_rna": DNA reverse complement with RNA output alphabet
            - "rna_to_dna": RNA reverse complement with DNA output alphabet
        
    Returns:
        Reverse complement of the sequence according to ``mode`` .
    """
    sequence = str(sequence).upper()

    if mode == "auto":
        mode = "rna_to_rna" if ("U" in sequence and "T" not in sequence) else "dna_to_dna"

    if mode == "dna_to_dna":
        complement_map = str.maketrans("ATCGU", "TAGCA")
    elif mode == "rna_to_rna":
        complement_map = str.maketrans("AUCGT", "UAGCA")
    elif mode == "dna_to_rna":
        complement_map = str.maketrans("ATCGU", "UAGCA")
    elif mode == "rna_to_dna":
        complement_map = str.maketrans("AUCGT", "TAGCA")
    else:
        raise ValueError(
            "mode must be one of: auto, dna_to_dna, rna_to_rna, dna_to_rna, rna_to_dna"
        )

    return sequence[::-1].translate(complement_map)


def sanitize_dna_sequence(
    sequence: str, replacement_base: str = "A", drop: bool = False
) -> str:
    """
    Normalize a sequence to uppercase DNA alphabet.

    Any non-ACGT characters are either removed (when ``drop=True``) or
    replaced with ``replacement_base``.
    """
    sequence = str(sequence).upper()
    if drop:
        return "".join(base for base in sequence if base in {"A", "C", "G", "T"})

    replacement_base = str(replacement_base).upper()
    if replacement_base not in {"A", "C", "G", "T"}:
        raise ValueError("replacement_base must be one of A/C/G/T")

    return "".join(base if base in {"A", "C", "G", "T"} else replacement_base for base in sequence)
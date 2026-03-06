"""Sequence manipulation utilities"""
from typing import Literal, Tuple

import numpy as np


def align_wt_mut_sequences(
    wt_sequence: str, 
    mut_sequence: str, 
    edit_position: int, 
    edit_length: int, 
    edit_type: int
) -> Tuple[str, str]:
    """
    Align the wild type and mutated sequences and add padding at mismatched location
    as a result of insertion or deletion
    
    Args:
        wt_sequence: Wild type sequence
        mut_sequence: Mutated sequence
        edit_position: Position of the edit (0-indexed)
        edit_length: Length of the edit
        edit_type: Type of edit (0=substitution, 1=insertion, 2=deletion)
        
    Returns:
        Tuple of (aligned_wt_sequence, aligned_mut_sequence)
    """
    l = len(wt_sequence)
    if edit_type == 1:  # insertion
        wt_sequence = wt_sequence[:edit_position] + 'N' * edit_length + wt_sequence[edit_position:]
    elif edit_type == 2:  # deletion
        mut_sequence = mut_sequence[:edit_position] + 'N' * edit_length + mut_sequence[edit_position:]
        
    # make sure the sequences are of the same length
    wt_sequence = wt_sequence[:l]
    mut_sequence = mut_sequence[:l]
    
    return wt_sequence, mut_sequence


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
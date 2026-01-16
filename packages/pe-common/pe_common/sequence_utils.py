"""Sequence manipulation utilities"""
from typing import Tuple


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

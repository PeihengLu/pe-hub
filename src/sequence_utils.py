from typing import Tuple


def align_wt_mut_sequences(wt_sequence: str, mut_sequence: str, edit_position: int, edit_length: int, edit_type: int) -> Tuple[str, str]:
    '''
    Align the wild type and mutated sequences and add padding at mismatched location
    as a result of insertion or deletion
    '''
    l = len(wt_sequence)
    if edit_type == 1: # insertion
        wt_sequence = wt_sequence[:edit_position] + 'N'*edit_length + wt_sequence[edit_position:]
    elif edit_type == 2: # deletion
        mut_sequence = mut_sequence[:edit_position] + 'N'*edit_length + mut_sequence[edit_position:]
        
    # make sure the sequences are of the same length
    wt_sequence = wt_sequence[:l]
    mut_sequence = mut_sequence[:l]
    
    return wt_sequence, mut_sequence
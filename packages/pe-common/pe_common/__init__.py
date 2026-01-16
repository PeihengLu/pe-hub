"""PE Common - Shared utilities for PE Database and PE Ensemble"""

__version__ = "0.1.0"

from .constants import PROJECT_ROOT, DATA_ROOT, MODEL_ROOT, DEVICE
from .sequence_utils import align_wt_mut_sequences, remove_padding
from .features import calculate_mfe, calculate_mt_wallace, calculate_gc_content

__all__ = [
    # Constants
    'PROJECT_ROOT',
    'DATA_ROOT',
    'MODEL_ROOT',
    'DEVICE',
    # Sequence utilities
    'align_wt_mut_sequences',
    'remove_padding',
    # Feature calculations
    'calculate_mfe',
    'calculate_mt_wallace',
    'calculate_gc_content',
]

"""PE Common - Shared utilities for PE Database and PE Ensemble"""

from typing import Any

__version__ = "0.1.0"

from .constants import PROJECT_ROOT, DATA_ROOT, MODEL_ROOT, DEVICE
from .sequence_utils import align_wt_mut_sequences, remove_padding
from .data_utils import build_test_mask_from_group_id
from .preprocessing import (
    has_columns,
    is_standardized_dataframe,
    standardized_to_pridict_dataframe,
    standardized_to_deepprime_features,
    ensure_schema,
)

__all__ = [
    # Constants
    "PROJECT_ROOT",
    "DATA_ROOT",
    "MODEL_ROOT",
    "DEVICE",
    # Sequence utilities
    "align_wt_mut_sequences",
    "remove_padding",
    "build_test_mask_from_group_id",
    # Preprocessing utilities
    "has_columns",
    "is_standardized_dataframe",
    "standardized_to_pridict_dataframe",
    "standardized_to_deepprime_features",
    "ensure_schema",
    # Feature calculations (lazy-loaded via __getattr__)
    "calculate_mfe",
    "calculate_mt_wallace",
    "calculate_gc_content",
]


def __getattr__(name: str) -> Any:
    if name in {"calculate_mfe", "calculate_mt_wallace", "calculate_gc_content"}:
        from .features import calculate_mfe, calculate_mt_wallace, calculate_gc_content

        return {
            "calculate_mfe": calculate_mfe,
            "calculate_mt_wallace": calculate_mt_wallace,
            "calculate_gc_content": calculate_gc_content,
        }[name]
    raise AttributeError(f"module 'pe_common' has no attribute '{name}'")

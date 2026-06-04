"""PE Common - Shared utilities for PE Database and PE Ensemble"""

from typing import Any

__version__ = "0.1.0"

from .constants import PROJECT_ROOT, DATA_ROOT, MODEL_ROOT, DEVICE
from .sequence_utils import align_wt_mut_sequences, remove_padding
from .data_utils import build_test_mask_from_group_id

# NOTE: standardized -> model-format conversion lives in the PE-DB service
# (services/pe-db/app/utils/convert_data.py) and is exposed via GET /api/filter.
# PE-Ensemble consumes model-format data from that endpoint; pe-common stays
# free of model-specific conversion logic.

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
    # Training utilities (lazy-loaded — requires torch)
    "EarlyStopping",
    "LightningTrainerConfig",
    "clip_gradients",
    "build_lr_scheduler",
    "build_group_kfold_indices",
    "ensure_lightning_available",
    "fit_lightning_module",
    "lightning_accelerator_from_device",
    "pearson_spearman",
    "run_supervised_training_loop",
    # Feature calculations (lazy-loaded)
    "calculate_mfe",
    "calculate_mt_wallace",
    "calculate_gc_content",
]

_TRAINING_EXPORTS = {
    "EarlyStopping",
    "LightningTrainerConfig",
    "clip_gradients",
    "build_lr_scheduler",
    "build_group_kfold_indices",
    "ensure_lightning_available",
    "fit_lightning_module",
    "lightning_accelerator_from_device",
    "pearson_spearman",
    "run_supervised_training_loop",
}

_FEATURE_EXPORTS = {
    "calculate_mfe",
    "calculate_mt_wallace",
    "calculate_gc_content",
}


def __getattr__(name: str) -> Any:
    if name in _TRAINING_EXPORTS:
        from . import training

        return getattr(training, name)
    if name in _FEATURE_EXPORTS:
        from .features import calculate_mfe, calculate_mt_wallace, calculate_gc_content

        return {
            "calculate_mfe": calculate_mfe,
            "calculate_mt_wallace": calculate_mt_wallace,
            "calculate_gc_content": calculate_gc_content,
        }[name]
    raise AttributeError(f"module 'pe_common' has no attribute '{name}'")

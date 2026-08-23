"""Registry for standardized -> model-format conversion functions."""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from .utils.convert_data import (
    standardized_to_deepprime_dataframe,
    standardized_to_oped_dataframe,
    standardized_to_optiprime_dataframe,
    standardized_to_pridict_dataframe,
)

ProgressCallback = Callable[[str], None]
FormatConverter = Callable[..., pd.DataFrame]

_FORMAT_CONVERTERS: dict[str, FormatConverter] = {}


def register_format(name: str, converter: FormatConverter) -> None:
    """Register a converter for ``name`` (case-insensitive)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Format name must not be empty.")
    _FORMAT_CONVERTERS[key] = converter


def unregister_format(name: str) -> None:
    key = name.strip().lower()
    _FORMAT_CONVERTERS.pop(key, None)


def is_format_registered(name: str) -> bool:
    return name.strip().lower() in _FORMAT_CONVERTERS


def get_format_converter(name: str) -> FormatConverter:
    key = name.strip().lower()
    if key not in _FORMAT_CONVERTERS:
        supported = sorted(_FORMAT_CONVERTERS.keys())
        raise ValueError(
            f"Unsupported target format: {name}. Supported: {supported}"
        )
    return _FORMAT_CONVERTERS[key]


def known_model_formats() -> frozenset[str]:
    """Format names that have a registered converter (excludes ``std``)."""
    return frozenset(_FORMAT_CONVERTERS.keys())


def known_output_formats() -> frozenset[str]:
    """All valid ``format`` values for export APIs, including ``std``."""
    return frozenset({"std", *_FORMAT_CONVERTERS.keys()})


def validate_output_format(name: str) -> str:
    """Normalize and validate an output format name."""
    key = name.strip().lower()
    if key not in known_output_formats():
        supported = sorted(known_output_formats())
        raise ValueError(f"Unsupported format: {name}. Supported: {supported}")
    return key


def convert_standardized(
    df: pd.DataFrame,
    target_format: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Convert a standardized dataframe to a registered model format."""
    key = validate_output_format(target_format)
    if key == "std":
        return df.copy()
    converter = get_format_converter(key)
    return converter(df, progress_callback=progress_callback)


def _register_builtin_formats() -> None:
    register_format("deepprime", standardized_to_deepprime_dataframe)
    register_format("pridict", standardized_to_pridict_dataframe)
    register_format("pridict2", standardized_to_pridict_dataframe)
    register_format("oped", standardized_to_oped_dataframe)
    register_format("optiprime", standardized_to_optiprime_dataframe)


_register_builtin_formats()

"""Canonical dataset keys for training hyperparameter presets."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

FilterScalar = Union[str, int]
FilterValue = Union[FilterScalar, Sequence[FilterScalar], None]


def normalize_segment(value: str) -> str:
    """Normalize a catalog segment for preset lookup."""
    return str(value).strip().lower().replace("-", "_")


def single_filter_value(value: FilterValue) -> Optional[str]:
    """Return a single filter value, or None when unset or ambiguous."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        item = value[0]
    else:
        item = value
    text = str(item).strip()
    return text if text else None


def dataset_preset_key(
    *,
    study: FilterValue = None,
    dataset: FilterValue = None,
    cell_line: FilterValue = None,
    pe_system: FilterValue = None,
) -> Optional[str]:
    """Build a preset lookup key such as ``pridict2/library2/hek293t/pe2``."""
    study_name = single_filter_value(study)
    dataset_name = single_filter_value(dataset)
    if not study_name or not dataset_name:
        return None

    base = f"{normalize_segment(study_name)}/{normalize_segment(dataset_name)}"
    cell = single_filter_value(cell_line)
    pe = single_filter_value(pe_system)
    if cell and pe:
        return f"{base}/{normalize_segment(cell)}/{normalize_segment(pe)}"
    return base


def candidate_preset_keys(
    *,
    study: FilterValue = None,
    dataset: FilterValue = None,
    cell_line: FilterValue = None,
    pe_system: FilterValue = None,
) -> list[str]:
    """Return preset keys from most specific to least specific."""
    keys: list[str] = []
    full = dataset_preset_key(
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )
    if full:
        keys.append(full)
    study_name = single_filter_value(study)
    dataset_name = single_filter_value(dataset)
    if study_name and dataset_name:
        base = f"{normalize_segment(study_name)}/{normalize_segment(dataset_name)}"
        if base not in keys:
            keys.append(base)
    return keys


def filters_from_request(request: Any) -> dict[str, FilterValue]:
    """Extract PE-DB filter fields from a training request."""
    return {
        "study": getattr(request, "study", None),
        "dataset": getattr(request, "dataset", None),
        "cell_line": getattr(request, "cell_line", None),
        "pe_system": getattr(request, "pe_system", None),
    }

"""Shared path/name normalization for PE-DB pipeline and catalog."""
from __future__ import annotations


def _normalize_name(value: str) -> str:
    """Normalize study/dataset/cell line/PE system names for filenames."""
    return str(value).strip().lower().replace("-", "_")


def normalize_name(value: str) -> str:
    return _normalize_name(value)


def canonical_dataset_name(value: str) -> str:
    """Catalog spelling: lowercase with hyphens."""
    return str(value).strip().lower().replace("_", "-")

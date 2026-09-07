"""Shared PE-DB filter field names and list-valued query coercion.

CLI flags, FastAPI query params, and ``filter_from_params`` all speak this
contract. Wire names must stay stable: PE Hub and ensemble ``GET /data/filter``
send these keys to PE-DB ``GET /api/filter``.
"""
from __future__ import annotations

import argparse
from typing import Any, Mapping, Optional

FILTER_LIST_FIELDS: tuple[str, ...] = (
    "study",
    "dataset",
    "cell_line",
    "pe_system",
    "edit_type",
    "edit_length",
    "edit_scope",
    "experimental_method",
    "target_context",
    "scaffold_name",
)

FILTER_RANGE_FIELDS: tuple[str, ...] = (
    "edit_efficiency_min",
    "edit_efficiency_max",
)


def coerce_list_param(value: Any) -> Optional[list[Any]]:
    """Normalize a scalar-or-list filter value; empty lists become ``None``."""
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    return [value]


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Append the catalog/edit filter flags used by ``pedb`` and ``peen``."""
    for name in FILTER_LIST_FIELDS:
        kwargs: dict[str, Any] = {"action": "append", "default": [], "dest": name}
        if name == "edit_length":
            kwargs["type"] = int
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    for name in FILTER_RANGE_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)


def filter_kwargs_from_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Extract filter fields from a mapping (argparse namespace via ``vars``)."""
    out: dict[str, Any] = {
        name: coerce_list_param(source.get(name)) for name in FILTER_LIST_FIELDS
    }
    for name in FILTER_RANGE_FIELDS:
        out[name] = source.get(name)
    return out


def filter_kwargs_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    return filter_kwargs_from_mapping(vars(args))

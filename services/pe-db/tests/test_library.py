"""Tests for PE-DB headless library API."""
from __future__ import annotations

import pytest

from app.library import (
    PeDbLibraryError,
    catalog_statistics,
    filter_data,
    filter_from_params,
    list_datasheets,
    list_datasets,
    list_output_formats,
    list_scaffolds,
    list_studies,
)


def test_list_output_formats_includes_builtins():
    formats = list_output_formats()
    assert "std" in formats
    assert "deepprime" in formats
    assert "oped" in formats


def test_filter_requires_split_strategy_when_format_set():
    with pytest.raises(PeDbLibraryError, match="split_strategy"):
        filter_data(format_="deepprime")


def test_filter_from_params_matches_filter_data():
    params = {
        "format": "std",
        "split_strategy": "none",
        "dataset": ["nonexistent-dataset-for-test"],
    }
    via_params = filter_from_params(params)
    direct = filter_data(
        format_="std",
        split_strategy="none",
        dataset=["nonexistent-dataset-for-test"],
    )
    assert via_params["status"] == "success"
    assert direct["status"] == "success"
    assert via_params.get("total_records", 0) == direct.get("total_records", 0)


def test_catalog_helpers_are_exported():
    assert callable(list_studies)
    assert callable(list_datasets)
    assert callable(list_datasheets)
    assert callable(list_scaffolds)
    assert callable(catalog_statistics)

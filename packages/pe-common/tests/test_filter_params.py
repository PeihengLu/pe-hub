"""Tests for shared PE-DB filter field helpers."""
from __future__ import annotations

import argparse
import inspect
import re

from pe_common.constants import PROJECT_ROOT
from pe_common.filter_params import (
    FILTER_LIST_DESCRIPTIONS,
    FILTER_LIST_FIELDS,
    FILTER_RANGE_DESCRIPTIONS,
    FILTER_RANGE_FIELDS,
    SPLIT_QUERY_DESCRIPTIONS,
    SPLIT_QUERY_FIELDS,
    CatalogFilterBody,
    CatalogFilterQuery,
    SplitExportQuery,
    add_filter_arguments,
    add_split_arguments,
    coerce_list_param,
    filter_kwargs_from_namespace,
    split_kwargs_from_mapping,
)


def test_coerce_list_param_none_and_empty():
    assert coerce_list_param(None) is None
    assert coerce_list_param([]) is None
    assert coerce_list_param("hek293") == ["hek293"]
    assert coerce_list_param(["a", "b"]) == ["a", "b"]


def test_add_filter_arguments_covers_list_fields():
    parser = argparse.ArgumentParser()
    add_filter_arguments(parser)
    args = parser.parse_args(
        ["--study", "deepprime", "--edit-length", "1", "--edit-efficiency-min", "0.2"]
    )
    kwargs = filter_kwargs_from_namespace(args)
    assert set(FILTER_LIST_FIELDS).issubset(kwargs)
    assert kwargs["study"] == ["deepprime"]
    assert kwargs["edit_length"] == [1]
    assert kwargs["dataset"] is None
    assert kwargs["edit_efficiency_min"] == 0.2
    assert kwargs["edit_efficiency_max"] is None


def test_add_split_arguments_covers_query_fields():
    parser = argparse.ArgumentParser()
    add_split_arguments(parser)
    args = parser.parse_args(["--split-strategy", "none", "--merge"])
    kwargs = split_kwargs_from_mapping(vars(args))
    assert set(SPLIT_QUERY_FIELDS) == set(kwargs)
    assert kwargs["split_strategy"] == "none"
    assert kwargs["merge"] is True
    assert kwargs["use_original_fold"] is False
    assert kwargs["split_random_state"] == 42


def test_query_and_body_models_follow_field_lists():
    assert set(FILTER_LIST_DESCRIPTIONS) == set(FILTER_LIST_FIELDS)
    assert set(FILTER_RANGE_DESCRIPTIONS) == set(FILTER_RANGE_FIELDS)
    assert set(SPLIT_QUERY_DESCRIPTIONS) == set(SPLIT_QUERY_FIELDS)
    catalog_fields = set(CatalogFilterQuery.model_fields)
    assert catalog_fields == set(FILTER_LIST_FIELDS) | set(FILTER_RANGE_FIELDS)
    assert set(SplitExportQuery.model_fields) == set(SPLIT_QUERY_FIELDS)
    assert set(FILTER_LIST_FIELDS) | set(FILTER_RANGE_FIELDS) <= set(
        CatalogFilterBody.model_fields
    )


def test_query_dependency_signature_lists_shared_fields():
    from pe_common.filter_params import query_dependency

    catalog = query_dependency(CatalogFilterQuery)
    names = list(inspect.signature(catalog).parameters)
    assert names == list(FILTER_LIST_FIELDS) + list(FILTER_RANGE_FIELDS)
    split = query_dependency(SplitExportQuery)
    assert list(inspect.signature(split).parameters) == list(SPLIT_QUERY_FIELDS)


def _ts_const_string_tuple(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"export const {name} = \[([^\]]*?)\] as const",
        source,
        re.S,
    )
    assert match, f"{name} not found in TypeScript exportAttributes"
    return tuple(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", match.group(1)))


def test_ts_filter_and_split_fields_match_python():
    source = (
        PROJECT_ROOT / "pe-hub" / "src" / "apps" / "database" / "config" / "exportAttributes.ts"
    ).read_text(encoding="utf-8")
    assert _ts_const_string_tuple(source, "FILTER_LIST_FIELDS") == FILTER_LIST_FIELDS
    assert _ts_const_string_tuple(source, "FILTER_RANGE_FIELDS") == FILTER_RANGE_FIELDS
    assert _ts_const_string_tuple(source, "SPLIT_QUERY_FIELDS") == SPLIT_QUERY_FIELDS

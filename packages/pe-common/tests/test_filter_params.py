"""Tests for shared PE-DB filter field helpers."""
from __future__ import annotations

import argparse

from pe_common.filter_params import (
    FILTER_LIST_FIELDS,
    add_filter_arguments,
    coerce_list_param,
    filter_kwargs_from_namespace,
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

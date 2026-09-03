"""Tests for HEK / HEK293T aliases and PRIDICT2 head selection."""
from __future__ import annotations

from pe_common.cell_lines import (
    canonical_cell_line,
    cell_line_alias_group,
    expand_cell_line_filter_values,
    pridict2_head_for_cell_line,
)


def test_hek_and_hek293t_are_the_same_line():
    assert canonical_cell_line("HEK") == "hek293t"
    assert canonical_cell_line("hek293t") == "hek293t"
    assert canonical_cell_line("HEK293T") == "hek293t"
    assert set(cell_line_alias_group("HEK")) == {"hek", "hek293t"}
    assert set(cell_line_alias_group("hek293t")) == {"hek", "hek293t"}


def test_filter_hek293t_also_matches_hek_sheets():
    assert expand_cell_line_filter_values(["hek293t"]) == ["hek", "hek293t"]
    assert expand_cell_line_filter_values(["HEK", "k562"]) == ["hek", "hek293t", "k562"]


def test_pridict2_head_matches_benchmark_cell():
    assert pridict2_head_for_cell_line("hek") == "HEK"
    assert pridict2_head_for_cell_line("HEK293T") == "HEK"
    assert pridict2_head_for_cell_line("k562") == "K562"
    assert pridict2_head_for_cell_line("k562mlh1dn") == "K562"
    assert pridict2_head_for_cell_line("hela") == "HEK"
    assert (
        pridict2_head_for_cell_line("k562mlh1dn", available_heads=["HEK", "K562", "K562MLH1dn"])
        == "K562MLH1dn"
    )

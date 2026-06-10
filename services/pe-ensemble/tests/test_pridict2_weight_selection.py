"""Tests for PRIDICT2 cell-type suffixed weight selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ROOT = REPO_ROOT / "services" / "pe-ensemble" / "weights" / "pridict2"


def _first_loadable_base_id() -> str:
    for entry in PRIDICT2ModelWrapper.list_available_weight_entries():
        weight_id = entry["id"]
        if "__HEK" in weight_id:
            return weight_id.rsplit("__", 1)[0]
    pytest.skip("No loadable PRIDICT2 multi-head weights in checkout")


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_list_available_weight_entries_include_cell_type_suffixes():
    entries = PRIDICT2ModelWrapper.list_available_weight_entries()
    assert entries
    assert all("__HEK" in entry["id"] or "__K562" in entry["id"] for entry in entries)
    assert all(entry.get("cell_type") in {"HEK", "K562"} for entry in entries)


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_resolve_weight_selection_parses_cell_type_suffix():
    base_id = _first_loadable_base_id()
    run_dir, cell_type = PRIDICT2ModelWrapper.resolve_weight_selection(f"{base_id}__HEK")
    assert run_dir.is_dir()
    assert cell_type == "HEK"


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_resolve_weight_selection_rejects_base_id_for_multi_head_run():
    base_id = _first_loadable_base_id()
    with pytest.raises(ValueError, match="cell-type head suffix"):
        PRIDICT2ModelWrapper.resolve_weight_selection(base_id)

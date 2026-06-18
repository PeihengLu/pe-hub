"""Tests for shared plugin manifest helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from pe_common.plugins import (
    activate_plugin,
    load_manifest,
    parse_manifest,
    plugin_status,
    validate_plugin_name,
)


def test_validate_plugin_name_rejects_builtin_collision():
    with pytest.raises(ValueError, match="built-in"):
        validate_plugin_name("deepprime")


def test_parse_dummy_manifest():
    manifest_path = (
        Path(__file__).resolve().parents[3]
        / "testdata"
        / "plugins"
        / "dummy_model"
        / "manifest.yaml"
    )
    manifest = parse_manifest(manifest_path)
    assert manifest.name == "dummy_model"
    assert manifest.model is not None
    assert manifest.model.pe_db_format == "dummy_model"
    assert manifest.format is not None
    assert manifest.format.output_columns == ("feature", "Efficiency")


def test_load_manifest_requires_matching_directory_name(tmp_path: Path):
    plugin_dir = tmp_path / "dummy_model"
    plugin_dir.mkdir()
    src = (
        Path(__file__).resolve().parents[3]
        / "testdata"
        / "plugins"
        / "dummy_model"
        / "manifest.yaml"
    )
    (plugin_dir / "manifest.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    load_manifest(plugin_dir)


def test_activate_plugin_writes_state(tmp_path: Path):
    plugin_dir = tmp_path / "dummy_model"
    plugin_dir.mkdir()
    src = (
        Path(__file__).resolve().parents[3]
        / "testdata"
        / "plugins"
        / "dummy_model"
        / "manifest.yaml"
    )
    (plugin_dir / "manifest.yaml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (plugin_dir / "convert.py").write_text("def convert(df):\n    return df\n", encoding="utf-8")
    state = activate_plugin(plugin_dir)
    assert state["status"] == "active"
    assert plugin_status(plugin_dir) == "active"

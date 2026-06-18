"""Tests for PE Database plugin loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.format_registry import convert_standardized, is_format_registered
from app.plugin_loader import load_active_plugins


@pytest.fixture()
def plugins_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    root = Path(__file__).resolve().parents[3] / "testdata" / "plugins"
    monkeypatch.setenv("PLUGINS_ROOT", str(root))
    return root


def _sample_standardized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wt_sequence": ["ACGTACGT", "TGCATGCA"],
            "mut_sequence": ["ACGTACGT", "TGCATGCA"],
            "editing_efficiency": [0.2, 0.5],
            "edit_len": [1, 3],
        }
    )


def test_load_dummy_plugin_registers_format(plugins_root: Path):
    loaded = load_active_plugins(plugins_root)
    assert "dummy_model" in loaded
    assert is_format_registered("dummy_model")

    converted = convert_standardized(_sample_standardized(), "dummy_model")
    assert list(converted.columns) == ["feature", "Efficiency"]
    assert converted["Efficiency"].tolist() == [0.2, 0.5]


def test_pending_plugin_is_not_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin_dir = tmp_path / "pending_plugin"
    plugin_dir.mkdir()
    src = (
        Path(__file__).resolve().parents[3]
        / "testdata"
        / "plugins"
        / "dummy_model"
        / "manifest.yaml"
    )
    text = src.read_text(encoding="utf-8").replace("dummy_model", "pending_plugin", 1)
    (plugin_dir / "manifest.yaml").write_text(text, encoding="utf-8")
    (plugin_dir / "convert.py").write_text(
        (
            Path(__file__).resolve().parents[3]
            / "testdata"
            / "plugins"
            / "dummy_model"
            / "convert.py"
        ).read_text(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLUGINS_ROOT", str(tmp_path))

    loaded = load_active_plugins(tmp_path)
    assert loaded == []
    assert "pending_plugin" not in loaded

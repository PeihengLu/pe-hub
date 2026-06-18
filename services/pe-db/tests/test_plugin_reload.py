"""Tests for PE-DB plugin reload endpoint."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.plugin_loader import _loaded_plugins, _quarantined_plugins, load_active_plugins, reload_active_plugins


@pytest.fixture()
def plugins_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    src = Path(__file__).resolve().parents[3] / "testdata" / "plugins"
    monkeypatch.setenv("PLUGINS_ROOT", str(src))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    return src


def test_reload_active_plugins(plugins_root: Path):
    loaded = reload_active_plugins(plugins_root)
    assert "dummy_model" in loaded

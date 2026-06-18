"""Training CLI loads active plugins into the model registry."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def dummy_plugins_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    from app.plugin_loader import _loaded_plugins, _quarantined_plugins

    plugins = Path(__file__).resolve().parents[3] / "testdata" / "plugins"
    monkeypatch.setenv("PLUGINS_ROOT", str(plugins))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    return plugins


def test_train_models_bootstrap_registers_active_plugins(dummy_plugins_root: Path):
    from app.train_models import _bootstrap_plugins
    from app.training.config import supported_models

    loaded = _bootstrap_plugins()
    assert "dummy_model" in loaded
    assert "dummy_model" in supported_models()


def test_build_parser_includes_plugin_models(dummy_plugins_root: Path):
    from app.train_models import _bootstrap_plugins, build_parser

    _bootstrap_plugins()
    parser = build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model")
    assert "dummy_model" in model_action.choices

"""Training CLI loads active plugins into the model registry."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture()
def dummy_plugins_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    from pe_ensemble.plugin_loader import _loaded_plugins, _quarantined_plugins

    plugins = Path(__file__).resolve().parents[3] / "testdata" / "plugins"
    monkeypatch.setenv("PLUGINS_ROOT", str(plugins))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    # Plugin loading mutates process-global registries, so undo it afterwards;
    # otherwise later tests see 'dummy_model' among the built-in models.
    yield plugins
    from pe_ensemble.plugin_loader import unregister_plugin

    for name in list(_loaded_plugins):
        unregister_plugin(name)
    _loaded_plugins.clear()
    _quarantined_plugins.clear()


def test_cli_bootstrap_registers_active_plugins(dummy_plugins_root: Path):
    from pe_ensemble.training.config import supported_models
    from pe_ensemble.cli import _bootstrap_plugins

    loaded = _bootstrap_plugins()
    assert "dummy_model" in loaded
    assert "dummy_model" in supported_models()


def test_train_parser_includes_plugin_models(dummy_plugins_root: Path):
    from pe_ensemble.cli import _bootstrap_plugins, build_parser

    _bootstrap_plugins()
    parser = build_parser()
    train_parser = _subparser(parser, "train")
    model_action = next(
        action for action in train_parser._actions if action.dest == "model"
    )
    assert "dummy_model" in model_action.choices


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices[name]

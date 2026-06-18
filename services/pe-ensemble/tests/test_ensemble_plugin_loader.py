"""Tests for PE Ensemble plugin loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def plugins_and_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    from app.plugin_loader import _loaded_plugins, _quarantined_plugins, load_active_plugins

    plugins = Path(__file__).resolve().parents[3] / "testdata" / "plugins"
    weights = tmp_path_factory.mktemp("ensemble_weights")
    monkeypatch.setenv("PLUGINS_ROOT", str(plugins))
    monkeypatch.setenv("WEIGHTS_ROOT", str(weights))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    load_active_plugins(plugins)
    return plugins


def test_load_dummy_plugin_registers_model_and_weights(plugins_and_weights: Path):
    from app.models import weights_registry
    from app.models.registry import model_registry
    from app.plugin_loader import load_active_plugins

    assert "dummy_model" in load_active_plugins(plugins_and_weights)
    assert model_registry.is_registered("dummy_model")

    spec = model_registry.get("dummy_model")
    assert spec.source == "plugin"
    assert spec.pe_db_format == "dummy_model"
    assert spec.weight_format == "dummy_state_dict"

    weight_ids = weights_registry.list_weight_ids("dummy_model")
    assert "dummy_base" in weight_ids


def test_dummy_wrapper_evaluate_round_trip(plugins_and_weights: Path):
    from app.models.registry import model_registry

    spec = model_registry.get("dummy_model")
    wrapper = spec.wrapper_class()
    test_df = pd.DataFrame({"feature": [1.0, 3.0], "Efficiency": [0.1, 0.3]})
    metrics = wrapper.evaluate(test_df, weights="dummy_base")
    assert "pearson" in metrics
    assert metrics["n_samples"] == 2

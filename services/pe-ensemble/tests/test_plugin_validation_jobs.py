"""Tests for async plugin validation jobs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_DUMMY_ROOT = Path(__file__).resolve().parents[3] / "testdata" / "plugins" / "dummy_model"


@pytest.fixture()
def plugins_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app.plugin_loader import _loaded_plugins, _quarantined_plugins

    root = tmp_path / "plugins"
    jobs = tmp_path / "validation_jobs"
    root.mkdir()
    jobs.mkdir()
    monkeypatch.setenv("PLUGINS_ROOT", str(root))
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path / "weights"))
    monkeypatch.setenv("PLUGIN_VALIDATION_JOBS_ROOT", str(jobs))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    return root


def _upload_dummy(plugins_root: Path) -> str:
    from app.plugins.manager import upload_plugin_bundle

    convert = (_DUMMY_ROOT / "convert.py").read_bytes()
    wrapper = (_DUMMY_ROOT / "wrapper.py").read_bytes()
    upload_plugin_bundle(
        name="async_dummy",
        version="0.1.0",
        display_name="Async Dummy",
        description="async validation test",
        wrapper_class="DummyModelWrapper",
        weight_format="dummy_state_dict",
        output_columns="feature,Efficiency",
        required_std_columns="edit_len,editing_efficiency",
        label_column="Efficiency",
        hyperparameters_json=json.dumps([{"name": "epochs", "type": "int", "default": 1}]),
        convert_bytes=convert,
        wrapper_bytes=wrapper,
    )
    return "async_dummy"


def test_queue_validation_completes_async(plugins_root: Path):
    from app.plugins.manager import queue_validation
    from app.plugins.validation_jobs import read_logs, wait_for_job

    _upload_dummy(plugins_root)
    created = queue_validation("async_dummy")
    assert created["status"] == "queued"

    manifest = wait_for_job(created["job_id"], timeout=120)
    assert manifest["status"] == "succeeded"
    assert manifest["result"]["validation_report"]["passed"]

    log_chunk, _ = read_logs(created["job_id"])
    assert "Validation finished" in log_chunk


def test_duplicate_validation_rejected(plugins_root: Path):
    from pe_common.plugins import PluginError

    from app.plugins.manager import queue_validation

    _upload_dummy(plugins_root)
    queue_validation("async_dummy")
    with pytest.raises(PluginError, match="already in progress"):
        queue_validation("async_dummy")

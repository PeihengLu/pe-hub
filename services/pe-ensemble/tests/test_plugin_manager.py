"""Tests for plugin upload, validation, and activation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_DUMMY_ROOT = Path(__file__).resolve().parents[3] / "testdata" / "plugins" / "dummy_model"


@pytest.fixture()
def plugins_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app.plugin_loader import _loaded_plugins, _quarantined_plugins

    root = tmp_path / "plugins"
    root.mkdir()
    monkeypatch.setenv("PLUGINS_ROOT", str(root))
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path / "weights"))
    _loaded_plugins.clear()
    _quarantined_plugins.clear()
    return root


def _dummy_bytes() -> tuple[bytes, bytes]:
    convert = (_DUMMY_ROOT / "convert.py").read_bytes()
    wrapper = (_DUMMY_ROOT / "wrapper.py").read_bytes()
    return convert, wrapper


def test_upload_validate_activate_flow(plugins_root: Path, monkeypatch: pytest.MonkeyPatch):
    from app.plugin_loader import load_active_plugins
    from app.plugins.manager import (
        activate_plugin_bundle,
        get_plugin,
        list_plugins,
        queue_validation,
        upload_plugin_bundle,
    )

    convert_bytes, wrapper_bytes = _dummy_bytes()
    weight_bytes = (_DUMMY_ROOT / "weights" / "dummy_base" / "weights.txt").read_bytes()

    uploaded = upload_plugin_bundle(
        name="upload_dummy",
        version="0.1.0",
        display_name="Upload Dummy",
        description="upload test",
        wrapper_class="DummyModelWrapper",
        weight_format="dummy_state_dict",
        output_columns="feature,Efficiency",
        required_std_columns="edit_len,editing_efficiency",
        label_column="Efficiency",
        hyperparameters_json=json.dumps([{"name": "epochs", "type": "int", "default": 1}]),
        weights_json=json.dumps([{"id": "dummy_base", "notes": "test"}]),
        convert_bytes=convert_bytes,
        wrapper_bytes=wrapper_bytes,
        weight_uploads=[("dummy_base", weight_bytes)],
    )
    assert uploaded["name"] == "upload_dummy"
    assert uploaded["status"] == "pending"

    plugins = list_plugins()
    assert any(entry["name"] == "upload_dummy" for entry in plugins)

    validation = queue_validation("upload_dummy")
    from app.plugins.validation_jobs import wait_for_job

    manifest = wait_for_job(validation["job_id"], timeout=120)
    assert manifest["status"] == "succeeded"
    assert manifest["result"]["validation_report"]["passed"]
    assert manifest["result"]["status"] == "pending"

    from app.plugins import manager as plugin_manager

    monkeypatch.setattr(
        plugin_manager,
        "notify_pe_db_plugin_reload",
        lambda: {"loaded": [], "count": 0},
    )
    activated = activate_plugin_bundle("upload_dummy")
    assert activated["status"] == "active"
    assert "upload_dummy" in activated["ensemble_loaded"]

    detail = get_plugin("upload_dummy")
    assert detail["status"] == "active"
    assert detail["validation_report"]["passed"]

    loaded = load_active_plugins(plugins_root)
    assert "upload_dummy" in loaded


def test_upload_with_manifest_file(plugins_root: Path):
    from app.plugins.manager import get_plugin, upload_plugin_bundle

    convert_bytes, wrapper_bytes = _dummy_bytes()
    manifest_bytes = (_DUMMY_ROOT / "manifest.yaml").read_bytes()

    uploaded = upload_plugin_bundle(
        manifest_bytes=manifest_bytes,
        convert_bytes=convert_bytes,
        wrapper_bytes=wrapper_bytes,
    )
    assert uploaded["name"] == "dummy_model"

    detail = get_plugin("dummy_model")
    manifest = detail["manifest"]
    assert manifest["model"]["class"] == "DummyModelWrapper"
    assert manifest["model"]["hyperparameters"]
    assert manifest["weights"]


def _dummy_zip_bytes() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for path in _DUMMY_ROOT.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            archive.writestr(path.relative_to(_DUMMY_ROOT).as_posix(), path.read_bytes())
    return buf.getvalue()


def test_upload_zip_bundle_only(plugins_root: Path):
    from app.plugins.manager import get_plugin, upload_plugin_bundle

    uploaded = upload_plugin_bundle(bundle_zip_bytes=_dummy_zip_bytes())
    assert uploaded["name"] == "dummy_model"

    detail = get_plugin("dummy_model")
    assert detail["manifest"]["name"] == "dummy_model"
    assert (plugins_root / "dummy_model" / "convert.py").is_file()
    assert (plugins_root / "dummy_model" / "wrapper.py").is_file()


def test_upload_manifest_name_mismatch(plugins_root: Path):
    from app.plugins.manager import upload_plugin_bundle

    convert_bytes, wrapper_bytes = _dummy_bytes()
    manifest_bytes = (_DUMMY_ROOT / "manifest.yaml").read_bytes()

    with pytest.raises(Exception, match="must match"):
        upload_plugin_bundle(
            name="other_name",
            manifest_bytes=manifest_bytes,
            convert_bytes=convert_bytes,
            wrapper_bytes=wrapper_bytes,
        )


def test_activate_requires_validation(plugins_root: Path):
    from app.plugins.manager import activate_plugin_bundle, upload_plugin_bundle

    convert_bytes, wrapper_bytes = _dummy_bytes()
    upload_plugin_bundle(
        name="no_validate",
        version="0.1.0",
        display_name="No Validate",
        description="test",
        wrapper_class="DummyModelWrapper",
        weight_format="dummy_state_dict",
        output_columns="feature,Efficiency",
        convert_bytes=convert_bytes,
        wrapper_bytes=wrapper_bytes,
    )
    with pytest.raises(Exception, match="validation"):
        activate_plugin_bundle("no_validate")


def test_delete_plugin_removes_directory(plugins_root: Path, monkeypatch: pytest.MonkeyPatch):
    from app.plugins.manager import (
        activate_plugin_bundle,
        delete_plugin_bundle,
        queue_validation,
        upload_plugin_bundle,
    )
    from app.plugins.validation_jobs import wait_for_job

    convert_bytes, wrapper_bytes = _dummy_bytes()
    upload_plugin_bundle(
        name="to_delete",
        version="0.1.0",
        display_name="Delete Me",
        description="test",
        wrapper_class="DummyModelWrapper",
        weight_format="dummy_state_dict",
        output_columns="feature,Efficiency",
        convert_bytes=convert_bytes,
        wrapper_bytes=wrapper_bytes,
    )
    manifest = wait_for_job(queue_validation("to_delete")["job_id"], timeout=120)
    assert manifest["status"] == "succeeded"
    from app.plugins import manager as plugin_manager

    monkeypatch.setattr(
        plugin_manager,
        "notify_pe_db_plugin_reload",
        lambda: {"loaded": [], "count": 0},
    )
    activate_plugin_bundle("to_delete")
    result = delete_plugin_bundle("to_delete")
    assert result["deleted"]
    assert not (plugins_root / "to_delete").exists()


def test_notify_pe_db_plugin_reload_handles_error(monkeypatch: pytest.MonkeyPatch):
    from app.plugins import manager as plugin_manager
    from app.plugins.manager import notify_pe_db_plugin_reload

    class FakeResponse:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    monkeypatch.setattr(
        plugin_manager.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )
    with pytest.raises(Exception, match="reload failed"):
        notify_pe_db_plugin_reload()

"""Tests for the plugin validation harness (Phase 3 correctness gate)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pe_common.plugin_validation import (
    _load_standardized_fixture,
    validate_plugin_directory,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLUGINS = _REPO_ROOT / "testdata" / "plugins"
_DUMMY = _PLUGINS / "dummy_model"
_BROKEN_CONVERT = _PLUGINS / "broken_convert"
_BROKEN_WRAPPER = _PLUGINS / "broken_wrapper"
_FIXTURE_CSV = _REPO_ROOT / "testdata" / "vendor_eval" / "standardized_small.csv"


@pytest.fixture()
def plugins_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "plugins"
    root.mkdir()
    monkeypatch.setenv("PLUGINS_ROOT", str(root))
    return root


def _copy_plugin(src: Path, dest_root: Path, name: str) -> Path:
    dest = dest_root / name
    shutil.copytree(src, dest)
    return dest


def test_standardized_fixture_loads_vendor_csv():
    df = _load_standardized_fixture()
    assert len(df) >= 2
    if _FIXTURE_CSV.is_file():
        assert "editing_efficiency" in df.columns


def test_dummy_plugin_passes_all_checks(plugins_root: Path):
    plugin_dir = _copy_plugin(_DUMMY, plugins_root, "dummy_model")
    report = validate_plugin_directory(plugin_dir)
    assert report.passed
    check_ids = {check.id for check in report.checks}
    assert check_ids == {
        "manifest_schema",
        "import_convert",
        "conversion_roundtrip",
        "import_wrapper",
        "interface_compliance",
        "train_smoke",
        "save_to_registry",
        "eval_smoke",
        "predict_smoke",
    }
    for check in report.checks:
        assert check.passed, check.id


def test_broken_convert_fails_import_convert(plugins_root: Path):
    plugin_dir = _copy_plugin(_BROKEN_CONVERT, plugins_root, "broken_convert")
    report = validate_plugin_directory(plugin_dir)
    assert not report.passed
    failed = {check.id for check in report.checks if not check.passed}
    assert "import_convert" in failed
    passed = {check.id for check in report.checks if check.passed}
    assert "manifest_schema" in passed


def test_broken_wrapper_fails_interface_compliance(plugins_root: Path):
    plugin_dir = _copy_plugin(_BROKEN_WRAPPER, plugins_root, "broken_wrapper")
    report = validate_plugin_directory(plugin_dir)
    assert not report.passed
    failed = {check.id for check in report.checks if not check.passed}
    assert "interface_compliance" in failed
    passed = {check.id for check in report.checks if check.passed}
    assert "import_convert" in passed
    assert "conversion_roundtrip" in passed


def test_validate_rejects_builtin_model_name(plugins_root: Path):
    plugin_dir = plugins_root / "deepprime"
    plugin_dir.mkdir()
    manifest_text = (_DUMMY / "manifest.yaml").read_text(encoding="utf-8").replace(
        "dummy_model", "deepprime"
    )
    (plugin_dir / "manifest.yaml").write_text(manifest_text, encoding="utf-8")
    (plugin_dir / "convert.py").write_text(
        (_DUMMY / "convert.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (plugin_dir / "wrapper.py").write_text(
        (_DUMMY / "wrapper.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="built-in"):
        validate_plugin_directory(plugin_dir)

"""Tests for PE-DB access in PE Ensemble (CLI in-process ``pe_db.library`` vs HTTP service)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PE_DB_ROOT = _REPO_ROOT / "services" / "pe-db"
_PE_COMMON = _REPO_ROOT / "packages" / "pe-common"


@pytest.fixture
def pe_db_cli_access(monkeypatch):
    """Enable pe-ensemble CLI pe-db access with pe-db on the import path."""
    from pe_ensemble.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", True)
    for entry in (_PE_DB_ROOT, _PE_COMMON):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    for name in list(sys.modules):
        if name == "pe_db" or name.startswith("pe_db."):
            del sys.modules[name]
    yield


def test_fetch_pe_db_filter_cli_in_process(pe_db_cli_access):
    from pe_ensemble.training.data import build_pe_db_filter_params, request_pe_db_filtered
    from pe_ensemble.training.schemas import SplitQueryParams

    params = build_pe_db_filter_params(
        model_format="std",
        split=SplitQueryParams(split_strategy="none"),
        dataset=["nonexistent-dataset-for-test"],
    )
    payload = request_pe_db_filtered(params)
    assert payload["status"] == "success"


def test_web_service_defaults_to_http(monkeypatch):
    from pe_ensemble.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", False)
    assert training_config.use_pe_db_library() is False
    assert training_config.pe_db_mode() == "http"


def test_enable_cli_pe_db_access_uses_in_process_pe_db(monkeypatch):
    from pe_ensemble.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", False)
    training_config.enable_cli_pe_db_access()
    assert training_config.use_pe_db_library() is True
    assert training_config.pe_db_mode() == "library"


def test_library_and_direct_imports_share_one_config_module():
    """pe_ensemble.library must not load a second copy of training.config.

    The old top-level ``app`` package collided with pe-db's ``app``, so library
    imported through ``pe_ensemble_service_app``. That produced two module
    objects for the same file; setting ``_use_pe_db_library`` on one copy did
    not update the other.
    """
    from pe_ensemble.library import supported_models
    from pe_ensemble.training import config as direct_config

    assert supported_models is direct_config.supported_models
    assert sys.modules["pe_ensemble.training.config"] is direct_config
    colliding = [
        name
        for name in sys.modules
        if name.startswith("app.training")
        or name.startswith("app.models")
        or name.startswith("pe_ensemble_service_app")
        or name == "pe_ensemble._bootstrap"
    ]
    assert colliding == []
    direct_config._use_pe_db_library = False
    direct_config.enable_cli_pe_db_access()
    assert direct_config.use_pe_db_library() is True

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
    from app.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", True)
    for entry in (_PE_DB_ROOT, _PE_COMMON):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    for name in list(sys.modules):
        if name == "pe_db" or name.startswith("pe_db.") or name.startswith("pe_db_service_app"):
            del sys.modules[name]
    yield


def test_fetch_pe_db_filter_cli_in_process(pe_db_cli_access):
    from app.training.data import build_pe_db_filter_params, request_pe_db_filtered
    from app.training.schemas import SplitQueryParams

    params = build_pe_db_filter_params(
        model_format="std",
        split=SplitQueryParams(split_strategy="none"),
        dataset=["nonexistent-dataset-for-test"],
    )
    payload = request_pe_db_filtered(params)
    assert payload["status"] == "success"


def test_web_service_defaults_to_http(monkeypatch):
    from app.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", False)
    assert training_config.use_pe_db_library() is False
    assert training_config.pe_db_mode() == "http"


def test_enable_cli_pe_db_access_uses_in_process_pe_db(monkeypatch):
    from app.training import config as training_config

    monkeypatch.setattr(training_config, "_use_pe_db_library", False)
    training_config.enable_cli_pe_db_access()
    assert training_config.use_pe_db_library() is True
    assert training_config.pe_db_mode() == "library"

"""Tests for PE-DB library mode in PE Ensemble."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PE_DB_ROOT = _REPO_ROOT / "services" / "pe-db"
_PE_COMMON = _REPO_ROOT / "packages" / "pe-common"


@pytest.fixture
def pe_db_library_mode(monkeypatch):
    """Enable library mode with pe-db on the import path."""
    monkeypatch.setenv("PE_DB_MODE", "library")
    for entry in (_PE_DB_ROOT, _PE_COMMON):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    # Drop cached pe_db modules so collision-safe bootstrap reloads.
    for name in list(sys.modules):
        if name == "pe_db" or name.startswith("pe_db.") or name.startswith("pe_db_service_app"):
            del sys.modules[name]
    yield


def test_fetch_pe_db_filter_library_mode(pe_db_library_mode):
    from app.training.data import build_pe_db_filter_params, request_pe_db_filtered
    from app.training.schemas import SplitQueryParams

    params = build_pe_db_filter_params(
        model_format="std",
        split=SplitQueryParams(split_strategy="none"),
        dataset=["nonexistent-dataset-for-test"],
    )
    payload = request_pe_db_filtered(params)
    assert payload["status"] == "success"


def test_pe_db_mode_default_is_http(monkeypatch):
    monkeypatch.delenv("PE_DB_MODE", raising=False)
    from app.training.config import pe_db_mode

    assert pe_db_mode() == "http"

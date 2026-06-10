"""Tests for PE-DB conversion progress tailing."""
from __future__ import annotations

import threading
import time

import pytest

from app.training.conversion_progress import pe_db_filter_progress
from pe_common.conversion_progress import append_progress


def test_pe_db_filter_progress_tails_messages(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PE_CONVERSION_PROGRESS_DIR", str(tmp_path))
    logged: list[str] = []
    started = threading.Event()

    def fake_request(params):
        started.set()
        token = params["progress_token"]
        append_progress(token, "Computing thermodynamic features: 1/10 (10%)")
        time.sleep(0.05)
        append_progress(token, "Computing RNA MFE features: 10/10 (100%)")
        return {"groups": [], "skipped": [], "total_records": 0}

    with pe_db_filter_progress({}, progress_log=logged.append, request_fn=fake_request) as payload:
        assert started.wait(timeout=2)
        assert payload["total_records"] == 0

    assert any("thermodynamic" in line for line in logged)
    assert any("100%" in line for line in logged)

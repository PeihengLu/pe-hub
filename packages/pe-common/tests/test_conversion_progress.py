"""Tests for conversion progress helpers."""
from __future__ import annotations

import pytest

from pe_common.conversion_progress import append_progress, clear_progress, progress_file_path


def test_progress_file_roundtrip(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PE_CONVERSION_PROGRESS_DIR", str(tmp_path))
    token = "abcd1234"
    clear_progress(token)
    append_progress(token, "Computing RNA MFE features: 10/100 (10%)")
    append_progress(token, "Computing RNA MFE features: 100/100 (100%)")
    text = progress_file_path(token).read_text(encoding="utf-8")
    assert "10/100" in text
    assert "100/100" in text
    clear_progress(token)
    assert not progress_file_path(token).exists()


def test_invalid_progress_token_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PE_CONVERSION_PROGRESS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Invalid progress token"):
        progress_file_path("../escape")

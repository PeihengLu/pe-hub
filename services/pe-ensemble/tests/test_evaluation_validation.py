"""Tests for evaluation request validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.schemas import EvaluationRequest
from app.models import weights_registry


def test_evaluation_request_requires_weights():
    with pytest.raises(ValidationError):
        EvaluationRequest(
            model_name="deepprime",
            benchmark_name="test-benchmark",
        )


def test_evaluation_request_rejects_blank_weights():
    with pytest.raises(ValidationError):
        EvaluationRequest(
            model_name="deepprime",
            benchmark_name="test-benchmark",
            weights="   ",
        )


def test_resolve_dir_rejects_unknown_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEIGHTS_ROOT", str(tmp_path / "weights"))
    weights_registry.rebuild_index()
    with pytest.raises(ValueError, match="does_not_exist"):
        weights_registry.resolve_dir("deepprime", "does_not_exist")

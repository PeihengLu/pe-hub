"""Runners must not special-case model names; wrappers own those hooks."""
from __future__ import annotations

from pathlib import Path

from pe_common.filter_params import FILTER_LIST_FIELDS, FILTER_RANGE_FIELDS
from pe_ensemble.training.schemas import TrainingRequest

_RUNNERS = (
    Path(__file__).resolve().parents[1] / "pe_ensemble" / "training" / "runner.py",
    Path(__file__).resolve().parents[1] / "pe_ensemble" / "evaluation" / "runner.py",
    Path(__file__).resolve().parents[1] / "pe_ensemble" / "ensemble" / "runner.py",
)


def test_train_eval_ensemble_runners_have_no_model_name_branches():
    for path in _RUNNERS:
        text = path.read_text(encoding="utf-8")
        assert 'if model_name == "oped"' not in text
        assert 'elif model_name == "oped"' not in text
        assert 'if model_name == "pridict2"' not in text
        assert 'elif model_name == "pridict2"' not in text


def test_training_request_exposes_catalog_filter_fields():
    names = set(TrainingRequest.model_fields)
    assert set(FILTER_LIST_FIELDS) <= names
    assert set(FILTER_RANGE_FIELDS) <= names

"""Tests for skipping evaluation jobs on non-exportable datasheets."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.evaluation.runner import _evaluation_skip_reason, execute_evaluation
from app.evaluation.schemas import EvaluationRequest
from app.training.data import ModelFormatFetchResult


def test_evaluation_skip_reason_for_non_standardizable_datasheet():
    fetch = ModelFormatFetchResult(
        df=pd.DataFrame(),
        skipped=[
            {
                "study": "pridict1",
                "dataset": "endogenous",
                "cell_line": "hek293t",
                "pe_system": "pe2",
                "reason": "dataset not standardizable",
            }
        ],
    )
    reason = _evaluation_skip_reason(fetch)
    assert reason is not None
    assert "pridict1/endogenous" in reason
    assert "dataset not standardizable" in reason


def test_evaluation_skip_reason_returns_none_when_nothing_matched():
    fetch = ModelFormatFetchResult(df=pd.DataFrame())
    assert _evaluation_skip_reason(fetch) is None


def test_execute_evaluation_marks_job_skipped_for_non_standardizable(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))

    request = EvaluationRequest(
        model_name="pridict2",
        benchmark_name="pridict1/endogenous",
        weights="pridict1_1__exp_2023-08-25_20-55-53__run_0",
        study="pridict1",
        dataset="endogenous",
        cell_line="hek293t",
        pe_system="pe2",
    )
    from app.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    empty_fetch = ModelFormatFetchResult(
        df=pd.DataFrame(),
        skipped=[
            {
                "study": "pridict1",
                "dataset": "endogenous",
                "cell_line": "hek293t",
                "pe_system": "pe2",
                "reason": "dataset not standardizable",
            }
        ],
    )

    with patch("app.evaluation.runner.fetch_model_format_result", return_value=empty_fetch):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["skipped"] is True
    assert "dataset not standardizable" in result["skip_reason"]
    manifest = get_job(job_id)
    assert manifest["status"] == "skipped"
    assert manifest["result"]["n_samples"] == 0


def test_execute_evaluation_still_fails_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="missing/data",
        weights="DeepPrime_base",
        study="missing",
        dataset="missing",
    )
    from app.evaluation.jobs import create_job, get_job
    from app.evaluation.runner import EvaluationError

    job_id = create_job(request)
    empty_fetch = ModelFormatFetchResult(df=pd.DataFrame())

    with patch("app.evaluation.runner.fetch_model_format_result", return_value=empty_fetch):
        with pytest.raises(EvaluationError, match="No test data resolved"):
            execute_evaluation(request, job_id=job_id, device_id="cpu")

    manifest = get_job(job_id)
    assert manifest["status"] == "failed"

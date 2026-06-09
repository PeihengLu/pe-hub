"""Tests for the evaluation job registry."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.evaluation.jobs import create_job, get_job, job_summary, mark_succeeded, wait_for_job
from app.evaluation.schemas import EvaluationRequest


@pytest.fixture()
def eval_jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "eval_jobs"
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(root))
    return root


def test_create_and_wait_for_eval_job(eval_jobs_root: Path):
    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="deepprime/library2",
        weights="DeepPrime_base",
    )
    job_id = create_job(request)
    manifest = get_job(job_id)
    assert manifest["status"] == "queued"
    assert manifest["benchmark_name"] == "deepprime/library2"

    mark_succeeded(job_id, {"metrics": {"pearson": 0.9}, "n_samples": 10})
    finished = wait_for_job(job_id, poll_interval=0.01)
    summary = job_summary(finished)
    assert summary.status == "succeeded"
    assert summary.weights_id == "DeepPrime_base"

"""Tests for the pe_ensemble.library job facade used by CLI and HTTP."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pe_ensemble import cli as peen_cli
from pe_ensemble import library
from pe_ensemble import main as ensemble_http
from pe_ensemble.library import TrainingRequest, ensure_job, get_job, queue_job


class _RecordingScheduler:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []

    def submit_training(self, job_id, request):
        self.submitted.append(("train", job_id))

    def submit_tuning(self, job_id, request):
        self.submitted.append(("tune", job_id))

    def submit_evaluation(self, job_id, request):
        self.submitted.append(("evaluate", job_id))

    def submit_ensemble(self, job_id, request):
        self.submitted.append(("ensemble", job_id))

    def device_snapshot(self):
        return []


@pytest.fixture
def jobs_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RecordingScheduler:
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(tmp_path / "jobs"))
    scheduler = _RecordingScheduler()
    monkeypatch.setattr("pe_ensemble.library.get_scheduler", lambda: scheduler)
    return scheduler


def _train_request() -> TrainingRequest:
    return TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        device="cpu",
    )


def test_queue_job_creates_record_and_submits(jobs_env: _RecordingScheduler):
    job_id, manifest = queue_job("train", _train_request())
    assert manifest["job_id"] == job_id
    assert manifest["status"] == "queued"
    assert jobs_env.submitted == [("train", job_id)]
    assert get_job("train", job_id)["model_name"] == "deepprime"


def test_ensure_job_reuses_existing_id(jobs_env: _RecordingScheduler):
    first = ensure_job("train", _train_request(), job_id="fixed-id")
    again = ensure_job("train", _train_request(), job_id="fixed-id")
    assert first == again == "fixed-id"
    assert jobs_env.submitted == []


def test_cli_and_http_do_not_import_job_stores_directly():
    cli_src = inspect.getsource(peen_cli)
    http_src = inspect.getsource(ensemble_http)
    assert "from pe_ensemble import library" in cli_src
    assert "from pe_ensemble import library" in http_src
    for name in (
        "pe_ensemble.training.jobs",
        "pe_ensemble.training.tune_jobs",
        "pe_ensemble.evaluation.jobs",
        "pe_ensemble.ensemble.jobs",
        "pe_ensemble.compute.device_scheduler",
        "pe_ensemble.compute.job_lifecycle",
    ):
        assert name not in cli_src, name
    for relative in (
        "from .training.jobs",
        "from .training.tune_jobs",
        "from .evaluation.jobs",
        "from .ensemble.jobs",
        "from .compute.device_scheduler",
        "from .compute.job_lifecycle",
    ):
        assert relative not in http_src, relative

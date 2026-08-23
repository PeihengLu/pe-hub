"""Tests for tuning job registry and scheduler integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.compute.device_scheduler import ComputeDeviceScheduler
from app.training.schemas import TrainingRequest
from app.training.tune_jobs import create_job, get_job, list_jobs, wait_for_job
from app.training.tuning_schemas import TuningRequest


@pytest.fixture()
def tune_jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "tune_jobs"
    monkeypatch.setenv("TUNING_JOBS_ROOT", str(root))
    return root


def _tuning_request(**overrides) -> TuningRequest:
    training = TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        study="deepprime",
        dataset="library2",
        device="cpu",
    )
    payload = {"training": training, "n_trials": 2, "no_write_preset": True}
    payload.update(overrides)
    return TuningRequest(**payload)


def test_create_tune_job_writes_manifest(tune_jobs_root: Path):
    job_id = create_job(_tuning_request())
    manifest = get_job(job_id)
    assert manifest["status"] == "queued"
    assert manifest["job_kind"] == "tune"
    assert manifest["n_trials"] == 2


def test_list_tune_jobs_newest_first(tune_jobs_root: Path):
    first = create_job(_tuning_request())
    second = create_job(_tuning_request())
    jobs = list_jobs(limit=10)
    assert [job["job_id"] for job in jobs[:2]] == [second, first]


def test_scheduler_runs_tune_job(tune_jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_execute(request, *, job_id=None, device_id=None):
        from app.training.tune_jobs import mark_succeeded

        mark_succeeded(
            job_id,
            {
                "study_name": "demo",
                "best_trial": 0,
                "best_value": 0.5,
                "preset_path": None,
            },
        )

    monkeypatch.setattr("app.compute.device_scheduler.execute_tuning", fake_execute)

    scheduler = ComputeDeviceScheduler()
    request = _tuning_request()
    job_id = create_job(request)
    scheduler.submit_tuning(job_id, request)
    manifest = wait_for_job(job_id, poll_interval=0.05, timeout=5)
    assert manifest["status"] == "succeeded"
    assert manifest["best_value"] == pytest.approx(0.5)

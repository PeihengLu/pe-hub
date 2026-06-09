"""Tests for killing and deleting compute jobs."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.compute.device_scheduler import ComputeDeviceScheduler
from app.compute.job_lifecycle import kill_and_remove_job
from app.training.jobs import create_job, get_job
from app.training.jobs import delete_job as delete_train_job
from app.training.schemas import TrainingRequest


@pytest.fixture()
def jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "jobs"
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(root))
    return root


def _training_request(**overrides) -> TrainingRequest:
    payload = {
        "model_name": "deepprime",
        "dataset_source": "pe-db",
        "dataset_name": "library2",
        "device": "cpu",
    }
    payload.update(overrides)
    return TrainingRequest(**payload)


def test_cancel_queued_job_and_delete(jobs_root: Path):
    scheduler = ComputeDeviceScheduler()
    job_id = create_job(_training_request())
    scheduler.submit_training(job_id, _training_request())

    assert get_job(job_id)["status"] in ("queued", "running")
    scheduler.cancel_job("train", job_id)

    kill_and_remove_job(
        "train",
        job_id,
        get_job=get_job,
        delete_job=delete_train_job,
    )
    assert not (jobs_root / job_id).exists()


def test_delete_terminal_job(jobs_root: Path):
    from app.training.jobs import mark_succeeded

    job_id = create_job(_training_request())
    mark_succeeded(job_id, {"weights_id": "w1", "weights_label": "W1"})

    kill_and_remove_job(
        "train",
        job_id,
        get_job=get_job,
        delete_job=delete_train_job,
    )
    assert not (jobs_root / job_id).exists()

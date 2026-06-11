"""Tests for killing and deleting compute jobs."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.compute import device_scheduler as device_scheduler_module
from app.compute.device_scheduler import ComputeDeviceScheduler, get_scheduler
from app.compute.job_lifecycle import begin_job_kill, finalize_job_kill, kill_and_remove_job
from app.training.jobs import create_job, get_job
from app.training.jobs import delete_job as delete_train_job
from app.training.schemas import TrainingRequest


@pytest.fixture()
def jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "jobs"
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(root))
    return root


@pytest.fixture()
def scheduler(monkeypatch: pytest.MonkeyPatch) -> ComputeDeviceScheduler:
    instance = ComputeDeviceScheduler()
    monkeypatch.setattr(device_scheduler_module, "_scheduler", instance)
    return instance


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
    job_id = create_job(_training_request())
    assert get_job(job_id)["status"] == "queued"

    kill_and_remove_job(
        "train",
        job_id,
        get_job=get_job,
        delete_job=delete_train_job,
    )
    assert not (jobs_root / job_id).exists()


def test_begin_kill_marks_running_job_stopping(jobs_root: Path, scheduler: ComputeDeviceScheduler):
    from app.training.jobs import mark_running

    job_id = create_job(_training_request())
    mark_running(job_id)
    queued = ("train", job_id)
    scheduler._job_device[queued] = "cpu:0"  # noqa: SLF001

    manifest = begin_job_kill("train", job_id, get_job=get_job)
    assert manifest is not None
    assert manifest["status"] == "stopping"
    assert get_job(job_id)["status"] == "stopping"

    from app.training.jobs import mark_cancelled

    mark_cancelled(job_id)
    finalize_job_kill(
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


def test_get_scheduler_returns_patched_instance(scheduler: ComputeDeviceScheduler):
    assert get_scheduler() is scheduler

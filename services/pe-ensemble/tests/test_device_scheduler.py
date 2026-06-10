"""Tests for per-device training queue."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.compute.device_scheduler import ComputeDeviceScheduler
from app.training.jobs import create_job, get_job, update_job, wait_for_job
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


def test_one_job_per_device(jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    gate = threading.Event()
    started = threading.Event()
    finished: list[str] = []

    def fake_execute(request, *, job_id=None, device_id=None):
        started.set()
        assert gate.wait(timeout=5)
        if job_id:
            update_job(job_id, status="succeeded", result={"weights_id": job_id})
            finished.append(job_id)

    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    job_a = create_job(_training_request(dataset_name="a"))
    job_b = create_job(_training_request(dataset_name="b"))

    scheduler.submit_training(job_a, _training_request(dataset_name="a"))
    assert started.wait(timeout=2)

    scheduler.submit_training(job_b, _training_request(dataset_name="b"))
    assert get_job(job_b)["queue_position"] == 1

    gate.set()
    deadline = time.time() + 5
    while len(finished) < 2 and time.time() < deadline:
        time.sleep(0.05)

    assert finished == [job_a, job_b]


def test_failed_job_updates_manifest(jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    from app.training.jobs import mark_running

    def fake_execute(request, *, job_id=None, device_id=None):
        if job_id:
            mark_running(job_id)
        raise ValueError("Unsupported PE system: pe2")

    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    job_id = create_job(_training_request())
    scheduler.submit_training(job_id, _training_request())
    wait_for_job(job_id, poll_interval=0.05, timeout=5)
    manifest = get_job(job_id)
    assert manifest["status"] == "failed"
    assert "Unsupported PE system" in (manifest.get("error") or "")


def test_explicit_device_assignment(jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    assigned: list[str] = []

    def fake_execute(request, *, job_id=None, device_id=None):
        assigned.append(device_id or "")
        if job_id:
            update_job(job_id, status="succeeded", result={"weights_id": job_id})

    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    job_id = create_job(_training_request(device="cpu"))
    scheduler.submit_training(job_id, _training_request(device="cpu"))
    wait_for_job(job_id, poll_interval=0.05, timeout=5)
    assert assigned == ["cpu"]


def test_shutdown_cancels_queued_jobs(jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    gate = threading.Event()

    def fake_execute(request, *, job_id=None, device_id=None):
        gate.wait(timeout=5)

    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    running_id = create_job(_training_request(dataset_name="running"))
    queued_id = create_job(_training_request(dataset_name="queued"))
    scheduler.submit_training(running_id, _training_request(dataset_name="running"))
    time.sleep(0.1)
    scheduler.submit_training(queued_id, _training_request(dataset_name="queued"))
    assert get_job(queued_id)["status"] == "queued"

    scheduler.shutdown(wait=False)
    assert get_job(queued_id)["status"] == "cancelled"

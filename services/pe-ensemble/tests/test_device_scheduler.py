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


def test_auto_assignment_waits_for_accelerator_not_cpu(
    jobs_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """Auto jobs must not use CPU while an accelerator is busy (batch benchmark safety)."""
    gate = threading.Event()
    assigned: list[str] = []

    def fake_list_device_ids(*, include_cpu: bool = True):
        if include_cpu:
            return ["cuda:0", "cpu"]
        return ["cuda:0"]

    def fake_list_accelerator_ids():
        return ["cuda:0"]

    def fake_execute(request, *, job_id=None, device_id=None):
        assigned.append(device_id or "")
        assert gate.wait(timeout=5)
        if job_id:
            update_job(job_id, status="succeeded", result={"weights_id": job_id})

    monkeypatch.setattr("app.compute.device_scheduler.list_device_ids", fake_list_device_ids)
    monkeypatch.setattr("app.compute.device_scheduler.list_accelerator_ids", fake_list_accelerator_ids)
    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    running_id = create_job(_training_request(device="auto", dataset_name="running"))
    queued_id = create_job(_training_request(device="auto", dataset_name="queued"))
    scheduler.submit_training(running_id, _training_request(device="auto", dataset_name="running"))
    time.sleep(0.1)
    scheduler.submit_training(queued_id, _training_request(device="auto", dataset_name="queued"))

    assert assigned == ["cuda:0"]
    assert get_job(queued_id)["queue_position"] == 1
    assert "cpu" not in assigned

    gate.set()
    wait_for_job(queued_id, poll_interval=0.05, timeout=5)
    assert assigned == ["cuda:0", "cuda:0"]


def test_device_snapshot_auto_jobs_not_counted_on_cpu(
    jobs_root: Path, monkeypatch: pytest.MonkeyPatch
):
    gate = threading.Event()

    def fake_list_device_ids(*, include_cpu: bool = True):
        if include_cpu:
            return ["cuda:0", "cpu"]
        return ["cuda:0"]

    def fake_list_accelerator_ids():
        return ["cuda:0"]

    def fake_execute(request, *, job_id=None, device_id=None):
        gate.wait(timeout=5)

    monkeypatch.setattr("app.compute.device_scheduler.list_device_ids", fake_list_device_ids)
    monkeypatch.setattr("app.compute.device_scheduler.list_accelerator_ids", fake_list_accelerator_ids)
    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    for index in range(3):
        job_id = create_job(_training_request(device="auto", dataset_name=f"job-{index}"))
        scheduler.submit_training(job_id, _training_request(device="auto", dataset_name=f"job-{index}"))

    snapshot = {item["device_id"]: item for item in scheduler.device_snapshot()}
    assert snapshot["cuda:0"]["queued_jobs"] == 2
    assert snapshot["cpu"]["queued_jobs"] == 0

    scheduler.shutdown(wait=False)


def test_device_snapshot_explicit_cpu_jobs_counted_on_cpu(
    jobs_root: Path, monkeypatch: pytest.MonkeyPatch
):
    gate = threading.Event()

    def fake_list_device_ids(*, include_cpu: bool = True):
        if include_cpu:
            return ["cuda:0", "cpu"]
        return ["cuda:0"]

    def fake_execute(request, *, job_id=None, device_id=None):
        gate.wait(timeout=5)

    monkeypatch.setattr("app.compute.device_scheduler.list_device_ids", fake_list_device_ids)
    monkeypatch.setattr("app.compute.device_scheduler.execute_training", fake_execute)

    scheduler = ComputeDeviceScheduler()
    running_id = create_job(_training_request(device="cpu", dataset_name="running"))
    queued_id = create_job(_training_request(device="cpu", dataset_name="queued"))
    scheduler.submit_training(running_id, _training_request(device="cpu", dataset_name="running"))
    time.sleep(0.1)
    scheduler.submit_training(queued_id, _training_request(device="cpu", dataset_name="queued"))

    snapshot = {item["device_id"]: item for item in scheduler.device_snapshot()}
    assert snapshot["cpu"]["queued_jobs"] == 1
    assert snapshot["cuda:0"]["queued_jobs"] == 0

    scheduler.shutdown(wait=False)


def test_auto_assignment_fails_without_accelerators(jobs_root: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.compute.device_scheduler.list_accelerator_ids", lambda: [])
    monkeypatch.setattr(
        "app.compute.device_scheduler.list_device_ids",
        lambda *, include_cpu=True: ["cpu"] if include_cpu else [],
    )

    scheduler = ComputeDeviceScheduler()
    job_id = create_job(_training_request(device="auto"))
    scheduler.submit_training(job_id, _training_request(device="auto"))
    manifest = get_job(job_id)
    assert manifest["status"] == "failed"
    assert "does not use CPU" in (manifest.get("error") or "")


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

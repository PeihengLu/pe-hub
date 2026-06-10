"""Tests for the training job registry and log streaming."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.jobs import (
    append_log,
    create_job,
    get_job,
    job_summary,
    list_jobs,
    mark_running,
    mark_succeeded,
    read_logs,
    update_job,
    wait_for_job,
)
from app.training.schemas import TrainingRequest


@pytest.fixture()
def jobs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "jobs"
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(root))
    return root


def test_create_job_and_logs(jobs_root: Path):
    request = TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        dataset=["library2"],
    )
    job_id = create_job(request)
    manifest = get_job(job_id)
    assert manifest["status"] == "queued"
    assert manifest["model_name"] == "deepprime"

    append_log(job_id, "line one")
    append_log(job_id, "line two")
    chunk, offset = read_logs(job_id, offset=0)
    assert "line one" in chunk
    assert "line two" in chunk

    chunk2, next_offset = read_logs(job_id, offset=offset)
    assert chunk2 == ""
    assert next_offset == offset

    mark_running(job_id)
    mark_succeeded(job_id, {"weights_id": "test-weights", "weights_label": "Test"})
    summary = job_summary(get_job(job_id))
    assert summary.status == "succeeded"
    assert summary.weights_id == "test-weights"

    request_path = jobs_root / job_id / "request.json"
    assert request_path.is_file()
    saved = json.loads(request_path.read_text(encoding="utf-8"))
    assert saved["dataset_name"] == "library2"


def test_list_jobs_sorted_by_created_at(jobs_root: Path):
    request = TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
    )
    older_id = create_job(request, job_id="older-job")
    newer_id = create_job(request, job_id="newer-job")
    update_job(older_id, created_at="2020-01-01T00:00:00Z")
    update_job(newer_id, created_at="2025-01-01T00:00:00Z")

    listed = list_jobs()
    assert [job["job_id"] for job in listed] == [newer_id, older_id]


def test_wait_for_job(jobs_root: Path):
    request = TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
    )
    job_id = create_job(request)
    mark_running(job_id)
    mark_succeeded(job_id, {"weights_id": "done", "weights_label": "Done"})
    manifest = wait_for_job(job_id, poll_interval=0.01)
    assert manifest["status"] == "succeeded"
    assert manifest["weights_id"] == "done"

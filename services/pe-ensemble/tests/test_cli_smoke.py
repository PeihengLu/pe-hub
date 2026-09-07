"""CLI smoke tests for every ``peen`` subcommand.

Heavy runners (train / tune / evaluate / ensemble) are stubbed. The tests
cover argument parsing, job creation, queue-only vs in-process dispatch, and
the listing commands. ``--list-devices`` is a global early-exit path and is
tested alongside the ``devices`` subcommand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pe_ensemble.cli import build_parser, main

pytestmark = pytest.mark.smoke

_OPED_WEIGHTS = "pegRNA_Model_Merged_saved.order3_decoder_weights"


class _QueueOnlyScheduler:
    def submit_training(self, *args, **kwargs):
        return None

    def submit_tuning(self, *args, **kwargs):
        return None

    def submit_evaluation(self, *args, **kwargs):
        return None

    def submit_ensemble(self, *args, **kwargs):
        return None

    def device_snapshot(self):
        return []


def _subcommand_paths(parser: argparse.ArgumentParser, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, subparser in action.choices.items():
            path = f"{prefix}{name}"
            nested = _subcommand_paths(subparser, prefix=f"{path} ")
            if nested:
                paths.update(nested)
            else:
                paths.add(path)
    return paths


def test_cli_subcommand_inventory():
    actual = _subcommand_paths(build_parser())
    declared = set(CLI_CASES)
    missing = actual - declared
    extra = declared - actual
    assert not missing, f"Add PE Ensemble CLI smoke coverage for: {sorted(missing)}"
    assert not extra, f"Remove stale PE Ensemble CLI smoke cases: {sorted(extra)}"


@pytest.fixture
def peen_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("TUNING_JOBS_ROOT", str(tmp_path / "tune_jobs"))
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setenv("ENSEMBLE_JOBS_ROOT", str(tmp_path / "ensemble_jobs"))
    monkeypatch.setenv("PLUGINS_ROOT", str(tmp_path / "plugins"))
    monkeypatch.setenv("TRAINING_PRESETS_ROOT", str(tmp_path / "presets"))
    (tmp_path / "plugins").mkdir()

    succeeded = {
        "status": "succeeded",
        "result": {"weights_id": "stub-weights", "metrics": {"pearson": 0.9}},
    }
    monkeypatch.setattr("pe_ensemble.cli.get_scheduler", lambda: _QueueOnlyScheduler())
    monkeypatch.setattr("pe_ensemble.cli.execute_tuning", lambda *a, **k: {"best_value": 0.1, "n_trials": 1})
    monkeypatch.setattr("pe_ensemble.cli.execute_evaluation", lambda *a, **k: succeeded["result"])
    monkeypatch.setattr("pe_ensemble.cli.execute_ensemble", lambda *a, **k: succeeded["result"])
    monkeypatch.setattr("pe_ensemble.cli.wait_for_train_job", lambda job_id, **k: succeeded)
    monkeypatch.setattr("pe_ensemble.cli.wait_for_tune_job", lambda job_id, **k: succeeded)
    monkeypatch.setattr("pe_ensemble.cli.wait_for_eval_job", lambda job_id, **k: succeeded)
    monkeypatch.setattr("pe_ensemble.cli.wait_for_ensemble_job", lambda job_id, **k: succeeded)

    from app.training import config as training_config

    previous = training_config.use_pe_db_library()
    yield tmp_path
    training_config._sync_pe_db_library_flag(previous)


def _run(argv: list[str]) -> int:
    return int(main(argv))


def _json_stdout(capsys: pytest.CaptureFixture[str]) -> object:
    captured = capsys.readouterr()
    text = captured.out.strip()
    for index, char in enumerate(text):
        if char in "{[":
            return json.loads(text[index:])
    raise AssertionError(f"No JSON in CLI stdout: {text!r}")


def _case_train(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            [
                "train",
                "--model",
                "deepprime",
                "--dataset-name",
                "library2",
                "--queue-only",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "job_id=" in out
    assert "Job queued." in out


def _case_tune(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            [
                "tune",
                "--model",
                "deepprime",
                "--dataset-name",
                "library2",
                "--n-trials",
                "1",
                "--no-write-preset",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = _json_stdout(capsys)
    assert payload["n_trials"] == 1


def _case_evaluate(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            [
                "evaluate",
                "--model",
                "deepprime",
                "--weights",
                "DeepPrime_base",
                "--queue-only",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "job_id=" in out


def _case_ensemble(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            [
                "ensemble",
                "--ensemble-name",
                "smoke",
                "--member",
                "deepprime:DeepPrime_base",
                "--member",
                f"oped:{_OPED_WEIGHTS}",
                "--queue-only",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "job_id=" in out


def _case_methods(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["methods"]) == 0
    out = capsys.readouterr().out
    assert "mean:" in out
    assert "weighted_mean:" in out


def _case_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["models"]) == 0
    payload = _json_stdout(capsys)
    names = {entry["name"] for entry in payload}
    assert "deepprime" in names


def _case_weights(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["weights", "--model", "deepprime"]) == 0
    payload = _json_stdout(capsys)
    ids = {entry["id"] for entry in payload}
    assert "DeepPrime_base" in ids


def _case_devices(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["devices"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()


def _case_jobs(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["jobs", "--kind", "train"]) == 0
    payload = _json_stdout(capsys)
    assert isinstance(payload, list)


def _case_logs(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            [
                "train",
                "--model",
                "deepprime",
                "--dataset-name",
                "library2",
                "--queue-only",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    queued = capsys.readouterr().out
    job_id = next(
        line.split("=", 1)[1].strip()
        for line in queued.splitlines()
        if line.startswith("job_id=")
    )
    assert _run(["logs", "--kind", "train", job_id]) == 0
    payload = _json_stdout(capsys)
    assert payload["job_id"] == job_id


CLI_CASES = {
    "train": _case_train,
    "tune": _case_tune,
    "evaluate": _case_evaluate,
    "ensemble": _case_ensemble,
    "methods": _case_methods,
    "models": _case_models,
    "weights": _case_weights,
    "devices": _case_devices,
    "jobs": _case_jobs,
    "logs": _case_logs,
}


@pytest.mark.parametrize("command", sorted(CLI_CASES))
def test_cli_command_smoke(
    command: str,
    peen_runtime,
    capsys: pytest.CaptureFixture[str],
):
    CLI_CASES[command](capsys)


def test_list_devices_early_exit(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert _run(["--list-devices"]) == 0
    assert capsys.readouterr().out.strip()


def test_train_runs_queued_job(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert (
        _run(
            [
                "train",
                "--model",
                "deepprime",
                "--dataset-name",
                "library2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = _json_stdout(capsys)
    assert payload["weights_id"] == "stub-weights"


def test_tune_queue_path(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert (
        _run(
            [
                "tune",
                "--model",
                "deepprime",
                "--dataset-name",
                "library2",
                "--n-trials",
                "1",
                "--no-write-preset",
                "--queue",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = _json_stdout(capsys)
    assert payload["weights_id"] == "stub-weights"


def test_evaluate_sync_path(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert (
        _run(
            [
                "evaluate",
                "--model",
                "deepprime",
                "--weights",
                "DeepPrime_base",
                "--sync",
                "--custom-benchmark",
                "--study",
                "deepprime",
                "--dataset",
                "library2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = _json_stdout(capsys)
    assert payload["metrics"]["pearson"] == 0.9


def test_ensemble_sync_path(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert (
        _run(
            [
                "ensemble",
                "--ensemble-name",
                "smoke",
                "--member",
                "deepprime:DeepPrime_base",
                "--member",
                f"oped:{_OPED_WEIGHTS}",
                "--sync",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = _json_stdout(capsys)
    assert payload["metrics"]["pearson"] == 0.9


def test_ensemble_requires_two_members(peen_runtime, capsys: pytest.CaptureFixture[str]):
    assert (
        _run(
            [
                "ensemble",
                "--ensemble-name",
                "smoke",
                "--member",
                "deepprime:DeepPrime_base",
            ]
        )
        == 1
    )
    assert "at least two" in capsys.readouterr().err.lower()


def test_jobs_kinds(peen_runtime, capsys: pytest.CaptureFixture[str]):
    for kind in ("train", "tune", "evaluate", "ensemble"):
        assert _run(["jobs", "--kind", kind]) == 0
        capsys.readouterr()


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main([])

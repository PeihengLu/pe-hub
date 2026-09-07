"""Tests for execute_tuning study helper."""
from __future__ import annotations

from pathlib import Path

import pytest

from pe_ensemble.training.runner import TrainingError
from pe_ensemble.training.schemas import TrainingRequest
from pe_ensemble.training.tune_study import execute_tuning
from pe_ensemble.training.tuning_schemas import TuningRequest


@pytest.fixture()
def tuning_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TUNING_STUDIES_ROOT", str(tmp_path / "studies"))
    monkeypatch.setenv("TRAINING_PRESETS_ROOT", str(tmp_path / "presets"))


def _request(**overrides) -> TuningRequest:
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


def test_execute_tuning_runs_trials(tuning_env, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_trial(request, *, suggested, register_weights=False):
        calls.append(dict(suggested))
        from pe_ensemble.training.tune_runner import TrialResult

        # Distinct metrics so Optuna has a stable best trial (maximize).
        return TrialResult(
            metric=float(suggested.get("epochs", 0)),
            hyperparameters=dict(suggested),
            train_result={},
        )

    monkeypatch.setattr("pe_ensemble.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "pe_ensemble.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {"epochs": trial.number + 1},
    )

    summary = execute_tuning(_request())
    assert summary["best_trial"] == 1
    assert len(calls) == 2


def test_execute_tuning_resumes_remaining_trials_only(
    tuning_env, monkeypatch: pytest.MonkeyPatch
):
    calls: list[dict] = []

    def fake_trial(request, *, suggested, register_weights=False):
        calls.append(dict(suggested))
        from pe_ensemble.training.tune_runner import TrialResult

        return TrialResult(
            metric=float(suggested.get("epochs", 0)),
            hyperparameters=dict(suggested),
            train_result={},
        )

    monkeypatch.setattr("pe_ensemble.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "pe_ensemble.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {"epochs": trial.number + 1},
    )

    first = execute_tuning(_request(n_trials=1, study_name="resume-cap"))
    assert len(calls) == 1
    second = execute_tuning(_request(n_trials=2, study_name="resume-cap"))
    assert len(calls) == 2
    third = execute_tuning(_request(n_trials=2, study_name="resume-cap"))
    assert len(calls) == 2
    assert second["best_trial"] == first["best_trial"] or second["best_trial"] == 1
    assert third["study_name"] == second["study_name"]


def test_execute_tuning_requires_dataset_key(tuning_env, monkeypatch: pytest.MonkeyPatch):
    from pe_ensemble.training.tune_runner import TrialResult

    def fake_trial(request, *, suggested, register_weights=False):
        return TrialResult(metric=0.1, hyperparameters=dict(suggested), train_result={})

    monkeypatch.setattr("pe_ensemble.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "pe_ensemble.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {},
    )

    request = _request()
    request.training.dataset = None
    request.training.study = None
    with pytest.raises(TrainingError, match="dataset preset key"):
        execute_tuning(request)


def test_execute_tuning_writes_merged_dataset_preset(tuning_env, monkeypatch: pytest.MonkeyPatch):
    from pe_ensemble.training.tune_runner import TrialResult

    def fake_trial(request, *, suggested, register_weights=False):
        return TrialResult(metric=0.1, hyperparameters=dict(suggested), train_result={})

    monkeypatch.setattr("pe_ensemble.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "pe_ensemble.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {"lr": 1e-4},
    )
    monkeypatch.setattr("pe_ensemble.training.tune_study.execute_training", lambda *args, **kwargs: {})

    # The shared helper suppresses preset writing; this test is about the
    # written preset, so opt back in (TRAINING_PRESETS_ROOT is a tmp dir).
    request = _request(no_write_preset=False)
    request.training.study = ["pridict1", "deepprime"]
    request.training.dataset = ["library1", "deepprime-clinvar"]
    request.training.cell_line = "hek293t"
    request.training.pe_system = "pe2"
    request.n_trials = 1

    summary = execute_tuning(request)
    assert summary["dataset_preset_key"] == (
        "pridict1/library1+deepprime/deepprime_clinvar/hek293t/pe2"
    )
    assert summary["preset_path"] is not None
    assert Path(summary["preset_path"]).is_file()


def test_register_best_weights_keeps_training_hyperparameter_mode(
    tuning_env, monkeypatch: pytest.MonkeyPatch
):
    from pe_ensemble.training.tune_runner import TrialResult

    captured: list[str] = []

    def fake_trial(request, *, suggested, register_weights=False):
        return TrialResult(metric=0.1, hyperparameters=dict(suggested), train_result={})

    def fake_train(request, *, device_id=None, register_weights=True):
        captured.append(request.hyperparameter_mode)
        return {"weights_id": "w1"}

    monkeypatch.setattr("pe_ensemble.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "pe_ensemble.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {"lr": 1e-4},
    )
    monkeypatch.setattr("pe_ensemble.training.tune_study.execute_training", fake_train)

    training = TrainingRequest(
        model_name="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        study="deepprime",
        dataset="library2",
        device="cpu",
        hyperparameter_mode="replace",
    )
    execute_tuning(
        TuningRequest(
            training=training,
            n_trials=1,
            no_write_preset=True,
            register_best_weights=True,
        )
    )
    assert captured == ["replace"]

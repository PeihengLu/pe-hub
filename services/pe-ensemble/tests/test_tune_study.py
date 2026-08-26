"""Tests for execute_tuning study helper."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.training.runner import TrainingError
from app.training.schemas import TrainingRequest
from app.training.tune_study import execute_tuning
from app.training.tuning_schemas import TuningRequest


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
        from app.training.tune_runner import TrialResult

        # Distinct metrics so Optuna has a stable best trial (maximize).
        return TrialResult(
            metric=float(suggested.get("epochs", 0)),
            hyperparameters=dict(suggested),
            train_result={},
        )

    monkeypatch.setattr("app.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "app.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {"epochs": trial.number + 1},
    )

    summary = execute_tuning(_request())
    assert summary["best_trial"] == 1
    assert len(calls) == 2


def test_execute_tuning_requires_dataset_key(tuning_env, monkeypatch: pytest.MonkeyPatch):
    from app.training.tune_runner import TrialResult

    def fake_trial(request, *, suggested, register_weights=False):
        return TrialResult(metric=0.1, hyperparameters=dict(suggested), train_result={})

    monkeypatch.setattr("app.training.tune_study.run_tuning_trial", fake_trial)
    monkeypatch.setattr(
        "app.training.tune_study.suggest_trial_hyperparameters",
        lambda model_name, trial: {},
    )

    request = _request()
    request.training.dataset = None
    request.training.study = None
    with pytest.raises(TrainingError, match="dataset preset key"):
        execute_tuning(request)

"""Tests for dataset hyperparameter presets and tuning helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.training.dataset_key import candidate_preset_keys, dataset_preset_key
from app.training.hyperparameter_presets import (
    SCHEDULER_KEYS,
    load_preset_bundle,
    resolve_hyperparameters,
    write_dataset_preset,
)
from app.training.schemas import TrainingRequest
from app.training.search_spaces import get_search_space, suggest_trial_hyperparameters
from app.training.tune_runner import extract_validation_metric


def test_dataset_preset_key_specificity():
    assert dataset_preset_key(
        study="pridict2",
        dataset="library2",
        cell_line="hek293t",
        pe_system="pe2",
    ) == "pridict2/library2/hek293t/pe2"
    assert dataset_preset_key(study="PRIDICT2", dataset="library-2") == "pridict2/library_2"


def test_candidate_preset_keys_most_specific_first():
    keys = candidate_preset_keys(
        study="pridict2",
        dataset="library2",
        cell_line="hek293t",
        pe_system="pe2",
    )
    assert keys == ["pridict2/library2/hek293t/pe2", "pridict2/library2"]


def test_resolve_hyperparameters_merge_order(tmp_path: Path):
    preset_file = tmp_path / "pridict2.yaml"
    preset_file.write_text(
        """
schema_version: 1
model: pridict2
defaults:
  batch_size: 128
  num_epochs: 20
datasets:
  pridict2/library2:
    hyperparameters:
      num_epochs: 25
      lr: 0.0003
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_hyperparameters(
        "pridict2",
        study="pridict2",
        dataset="library2",
        user_overrides={"batch_size": 256},
        preset_root=tmp_path,
    )
    assert resolved.preset_key == "pridict2/library2"
    assert resolved.hyperparameters["num_epochs"] == 25
    assert resolved.hyperparameters["lr"] == 0.0003
    assert resolved.hyperparameters["batch_size"] == 256


def test_resolve_hyperparameters_replace_skips_preset(tmp_path: Path):
    preset_file = tmp_path / "oped.yaml"
    preset_file.write_text(
        """
model: oped
defaults:
  lr: 0.01
datasets:
  deeppe/deeppe_ht:
    hyperparameters:
      lr: 0.02
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_hyperparameters(
        "oped",
        study="deeppe",
        dataset="deeppe_ht",
        user_overrides={"lr": 0.0005},
        mode="replace",
        preset_root=tmp_path,
    )
    assert resolved.preset_key is None
    assert resolved.hyperparameters["lr"] == 0.0005


def test_scheduler_keys_excluded_from_preset_write(tmp_path: Path):
    path = tmp_path / "oped.yaml"
    write_dataset_preset(
        path,
        model_name="oped",
        dataset_key="deeppe/deeppe_ht",
        hyperparameters={
            "lr": 0.001,
            "scheduler": "cosine",
            "scheduler_kwargs": {"t_max": 10},
        },
        provenance={"source": "test"},
    )
    bundle = load_preset_bundle("oped", root=tmp_path)
    entry = bundle["datasets"]["deeppe/deeppe_ht"]["hyperparameters"]
    assert "scheduler" not in entry
    assert "scheduler_kwargs" not in entry
    assert entry["lr"] == 0.001
    assert "scheduler" in SCHEDULER_KEYS


def test_search_space_excludes_scheduler_keys():
    space = get_search_space("oped")
    assert "scheduler" not in space.params
    assert "scheduler_kwargs" not in space.params


def test_suggest_trial_hyperparameters_no_scheduler():
    class _Trial:
        def suggest_float(self, name, low, high, log=False):
            return 0.001

        def suggest_int(self, name, low, high):
            return 50

        def suggest_categorical(self, name, choices):
            return choices[0]

    suggested = suggest_trial_hyperparameters("oped", _Trial())
    assert "scheduler" not in suggested
    assert "scheduler_kwargs" not in suggested
    assert suggested["load_pretrained"] is False
    assert suggested["hidden_size"] == [1024, 1024, 1024]


def test_extract_validation_metric_oped():
    metric = extract_validation_metric(
        "oped",
        {"result": {"val_spearman": 0.42}},
    )
    assert metric == pytest.approx(0.42)


def test_extract_validation_metric_deepprime_neg_loss():
    metric = extract_validation_metric(
        "deepprime",
        {"result": {"model_summaries": [{"best_val_loss": 0.2}]}},
    )
    assert metric == pytest.approx(-0.2)


def test_extract_validation_metric_pridict2():
    metric = extract_validation_metric(
        "pridict2",
        {"result": {"validation_metrics": {"averageedited_spearman": 0.55}}},
    )
    assert metric == pytest.approx(0.55)


def test_training_request_hyperparameter_mode_default():
    request = TrainingRequest(
        model_name="oped",
        dataset_source="pe-db",
        dataset_name="test",
    )
    assert request.hyperparameter_mode == "merge"

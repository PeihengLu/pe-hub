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


def test_resolve_hyperparameters_local_overlays_shipped(tmp_path: Path):
    shipped = tmp_path / "shipped"
    local = tmp_path / "local"
    shipped.mkdir()
    local.mkdir()
    (shipped / "pridict2.yaml").write_text(
        """
model: pridict2
defaults:
  batch_size: 128
  lr: 0.0001
datasets:
  pridict2/library2:
    hyperparameters:
      num_epochs: 20
      lr: 0.0002
""".strip(),
        encoding="utf-8",
    )
    (local / "pridict2.yaml").write_text(
        """
model: pridict2
defaults:
  batch_size: 64
datasets:
  pridict2/library2:
    hyperparameters:
      num_epochs: 30
""".strip(),
        encoding="utf-8",
    )

    resolved = resolve_hyperparameters(
        "pridict2",
        study="pridict2",
        dataset="library2",
        shipped_root=shipped,
        local_root=local,
    )
    assert resolved.preset_key == "pridict2/library2"
    assert resolved.preset_source == "local_preset:pridict2/library2"
    assert resolved.hyperparameters["batch_size"] == 64  # local defaults
    assert resolved.hyperparameters["num_epochs"] == 30  # local dataset
    assert resolved.hyperparameters["lr"] == 0.0002  # shipped dataset (local omitted)


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


def test_pridict2_search_space_omits_derived_architecture_knobs():
    space = get_search_space("pridict2")
    assert "assemb_opt" not in space.fixed
    assert "assemb_opt" not in space.params
    assert "annot_embed" not in space.fixed
    assert "annot_embed" not in space.params
    assert "z_dim" not in space.params
    assert space.fixed["load_pretrained"] is False

    class _Trial:
        def suggest_float(self, name, low, high, log=False):
            return 0.001

        def suggest_int(self, name, low, high):
            return 20

        def suggest_categorical(self, name, choices):
            return 128 if name == "embed_dim" else choices[0]

    suggested = suggest_trial_hyperparameters("pridict2", _Trial())
    assert suggested["embed_dim"] == 128
    assert "assemb_opt" not in suggested
    assert "annot_embed" not in suggested
    assert "z_dim" not in suggested


def test_materialize_hyperparameters_applies_fixed_and_oped_aliases():
    from app.training.search_spaces import materialize_hyperparameters

    oped = materialize_hyperparameters(
        "oped",
        {
            "ffn_dim": 1024,
            "encoder_layers": 4,
            "embedding_size": 64,
            "nhead": 8,
            "dropout": 0.2,
            "lr": 1e-4,
        },
    )
    assert oped["load_pretrained"] is False
    assert oped["hidden_size"] == [1024, 1024, 1024]
    assert oped["num_encoder_layers"] == [4, 4, 4]
    assert oped["drop_out"] == 0.2
    assert "ffn_dim" not in oped
    assert "encoder_layers" not in oped

    deepprime = materialize_hyperparameters(
        "deepprime",
        {"hidden_size": 256, "num_layers": 2, "epochs": 10},
    )
    assert deepprime["load_pretrained"] is False
    assert deepprime["hidden_size"] == 256


def test_pridict2_seqlevel_featdim_uses_datatensor_colnames():
    from types import SimpleNamespace

    from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

    dtensor = SimpleNamespace(
        seqlevel_feat_colnames=["a"] * 18,
        seqlevel_feat=None,
    )
    assert PRIDICT2ModelWrapper._seqlevel_featdim_from_datatensor(dtensor) == 18


def test_extract_validation_metric_oped():
    metric = extract_validation_metric(
        "oped",
        {"result": {"val_spearman": 0.42}},
    )
    assert metric == pytest.approx(0.42)


def test_extract_validation_metric_oped_prefers_cv_mean():
    metric = extract_validation_metric(
        "oped",
        {
            "result": {
                "val_spearman": 0.99,
                "cross_validation": {"mean_val_spearman": 0.4},
            }
        },
    )
    assert metric == pytest.approx(0.4)


def test_extract_validation_metric_deepprime_neg_loss():
    metric = extract_validation_metric(
        "deepprime",
        {"result": {"model_summaries": [{"best_val_loss": 0.2}]}},
    )
    assert metric == pytest.approx(-0.2)


def test_extract_validation_metric_deepprime_prefers_cv_mean():
    metric = extract_validation_metric(
        "deepprime",
        {
            "result": {
                "model_summaries": [{"best_val_loss": 0.01}],
                "cross_validation": {
                    "mean_best_val_loss": 0.5,
                    "folds": [
                        {"best_val_loss": 0.4},
                        {"best_val_loss": 0.6},
                    ],
                },
            }
        },
    )
    assert metric == pytest.approx(-0.5)


def test_extract_validation_metric_pridict2():
    metric = extract_validation_metric(
        "pridict2",
        {"result": {"validation_metrics": {"averageedited_spearman": 0.55}}},
    )
    assert metric == pytest.approx(0.55)


def test_extract_validation_metric_pridict2_prefers_cv_mean():
    metric = extract_validation_metric(
        "pridict2",
        {
            "result": {
                "validation_metrics": {"averageedited_spearman": 0.99},
                "cross_validation": [
                    {"metrics": {"averageedited_spearman": 0.2}},
                    {"metrics": {"averageedited_spearman": 0.4}},
                ],
            }
        },
    )
    assert metric == pytest.approx(0.3)


def test_training_request_hyperparameter_mode_default():
    request = TrainingRequest(
        model_name="oped",
        dataset_source="pe-db",
        dataset_name="test",
    )
    assert request.hyperparameter_mode == "merge"

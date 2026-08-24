"""Tests for pe-ensemble CLI request builders and package imports."""
from __future__ import annotations

import argparse

import pytest

from app.ensemble.combine import COMBINE_METHODS
from app.training.config import supported_models
from pe_ensemble.cli import (
    build_ensemble_request,
    build_evaluation_request,
    build_training_request,
    build_tuning_request,
    parse_ensemble_member,
)


def test_supported_models_importable():
    assert "deepprime" in supported_models()


def test_build_split_cv_clears_holdout_pcts():
    args = argparse.Namespace(
        split_strategy="cv",
        train_pct=None,
        val_pct=None,
        test_pct=None,
        cv_folds=3,
        use_original_fold=False,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
    )
    from pe_ensemble.cli import _build_split

    split = _build_split(args)
    assert split.split_strategy == "cv"
    assert split.cv_folds == 3
    assert split.train_pct is None
    assert split.val_pct is None
    assert split.test_pct is None


def test_build_split_holdout3_applies_defaults():
    args = argparse.Namespace(
        split_strategy="holdout_3",
        train_pct=None,
        val_pct=None,
        test_pct=None,
        cv_folds=None,
        use_original_fold=False,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
    )
    from pe_ensemble.cli import _build_split

    split = _build_split(args)
    assert split.train_pct == 0.7
    assert split.val_pct == 0.15
    assert split.test_pct == 0.15


def test_combine_methods_exported():
    assert "mean" in COMBINE_METHODS
    assert "weighted_mean" in COMBINE_METHODS


def test_build_training_request_merges_architecture_flags():
    args = argparse.Namespace(
        model="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        study=[],
        dataset=["library2"],
        cell_line=["HEK293T"],
        pe_system=["PE2max"],
        edit_type=[],
        edit_length=[],
        edit_scope=[],
        experimental_method=[],
        target_context=[],
        scaffold_name=[],
        edit_efficiency_min=None,
        edit_efficiency_max=None,
        split_strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
        cv_folds=None,
        use_original_fold=False,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
        hyperparameters_json='{"epochs": 1}',
        hyperparameter_mode="merge",
        pretrained_weights=None,
        model_kwargs_json=None,
        notes=None,
        device="cpu",
        dp_hidden_size=64,
        dp_num_layers=1,
        oped_embedding_size=None,
        oped_ffn_dim=None,
        oped_encoder_layers=None,
        oped_nhead=None,
        oped_dropout=None,
        pridict2_embed_dim=None,
        pridict2_z_dim=None,
        pridict2_num_hidden_layers=None,
        pridict2_dropout=None,
    )
    request = build_training_request(args)
    assert request.model_name == "deepprime"
    assert request.hyperparameters is not None
    assert request.hyperparameters["hidden_size"] == 64
    assert request.hyperparameters["epochs"] == 1


def test_build_tuning_request_wraps_training():
    args = argparse.Namespace(
        model="deepprime",
        dataset_source="pe-db",
        dataset_name="library2",
        study=["deepprime"],
        dataset=["library2"],
        cell_line=[],
        pe_system=[],
        edit_type=[],
        edit_length=[],
        edit_scope=[],
        experimental_method=[],
        target_context=[],
        scaffold_name=[],
        edit_efficiency_min=None,
        edit_efficiency_max=None,
        split_strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
        cv_folds=None,
        use_original_fold=False,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
        fixed_hyperparameters_json=None,
        model_kwargs_json=None,
        notes=None,
        device="cpu",
        n_trials=3,
        study_name=None,
        study_storage=None,
        write_preset=None,
        no_write_preset=True,
        register_best_weights=False,
    )
    request = build_tuning_request(args)
    assert request.n_trials == 3
    assert request.training.model_name == "deepprime"
    assert request.no_write_preset is True


def test_parse_ensemble_member():
    member = parse_ensemble_member("deepprime:weights-a:0.5")
    assert member.model_name == "deepprime"
    assert member.weights == "weights-a"
    assert member.member_weight == pytest.approx(0.5)


def test_build_evaluation_request_auto_benchmark():
    args = argparse.Namespace(
        model="deepprime",
        weights="test-weights",
        benchmark_name=None,
        study=[],
        dataset=[],
        cell_line=[],
        pe_system=[],
        edit_type=[],
        edit_length=[],
        edit_scope=[],
        experimental_method=[],
        target_context=[],
        scaffold_name=[],
        edit_efficiency_min=None,
        edit_efficiency_max=None,
        split_strategy="holdout_2",
        train_pct=0.8,
        val_pct=None,
        test_pct=0.2,
        cv_folds=None,
        use_original_fold=True,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
        custom_benchmark=False,
        allow_data_leak=False,
        device="cpu",
    )
    request = build_evaluation_request(args)
    assert request.auto_training_benchmark is True
    assert request.weights == "test-weights"


def test_build_ensemble_request_requires_members():
    args = argparse.Namespace(
        ensemble_name="demo",
        combine="mean",
        combine_options_json=None,
        member=["deepprime:w1", "oped:w2"],
        study=[],
        dataset=[],
        cell_line=[],
        pe_system=[],
        edit_type=[],
        edit_length=[],
        edit_scope=[],
        experimental_method=[],
        target_context=[],
        scaffold_name=[],
        edit_efficiency_min=None,
        edit_efficiency_max=None,
        split_strategy="holdout_2",
        train_pct=0.8,
        val_pct=None,
        test_pct=0.2,
        cv_folds=None,
        use_original_fold=True,
        original_fold_test_value=-1.0,
        split_random_state=42,
        merge=False,
        device="cpu",
    )
    request = build_ensemble_request(args)
    assert request.combine == "mean"
    assert len(request.members) == 2

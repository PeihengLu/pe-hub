"""Tests that split requests are rejected at the API boundary.

``pe_common.splits.SplitConfig`` performs equivalent checks, but only after
PE-DB has loaded and converted the data. Validating on the request model turns
a malformed split into an immediate 422 instead of a queued job that dies
minutes later.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.training.schemas import SplitQueryParams, TrainingRequest, default_training_split


class TestValidRequests:
    def test_default_training_split_is_accepted(self):
        split = default_training_split()
        assert split.split_strategy == "holdout_3"
        assert pytest.approx(1.0) == split.train_pct + split.val_pct + split.test_pct

    def test_strategy_none_needs_no_fractions(self):
        assert SplitQueryParams().split_strategy == "none"

    def test_holdout_2_accepts_complementary_fractions(self):
        split = SplitQueryParams(split_strategy="holdout_2", train_pct=0.8, test_pct=0.2)
        assert split.train_pct == 0.8

    def test_cv_accepts_fold_count(self):
        assert SplitQueryParams(split_strategy="cv", cv_folds=5).cv_folds == 5

    def test_cv_accepts_outer_test_holdout(self):
        split = SplitQueryParams(split_strategy="cv", cv_folds=5, test_pct=0.15)
        assert split.test_pct == 0.15


class TestInvalidRequests:
    @pytest.mark.parametrize(
        "params, expected",
        [
            (
                dict(split_strategy="holdout_3", train_pct=0.7, val_pct=0.15, test_pct=0.30),
                "sum to 1.0",
            ),
            (dict(split_strategy="holdout_3", train_pct=0.7, val_pct=0.15), "requires test_pct"),
            (dict(split_strategy="holdout_2", train_pct=0.8), "requires test_pct"),
            (dict(split_strategy="holdout_2", train_pct=1.0, test_pct=0.2), "between 0 and 1"),
            (dict(split_strategy="cv"), "requires cv_folds"),
            (dict(split_strategy="cv", cv_folds=1), "cv_folds must be >= 2"),
            (dict(split_strategy="cv", cv_folds=5, test_pct=1.5), "between 0 and 1"),
        ],
    )
    def test_rejected(self, params, expected):
        with pytest.raises(ValidationError, match=expected):
            SplitQueryParams(**params)

    def test_training_request_surfaces_split_errors(self):
        with pytest.raises(ValidationError, match="sum to 1.0"):
            TrainingRequest(
                model_name="deepprime",
                dataset_source="pe-db",
                dataset_name="demo",
                split=dict(
                    split_strategy="holdout_3", train_pct=0.5, val_pct=0.1, test_pct=0.1
                ),
            )

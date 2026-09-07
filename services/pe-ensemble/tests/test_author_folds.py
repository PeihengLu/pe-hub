"""Tests for vendor provenance fold predicates."""
from __future__ import annotations

from app.models.author_folds import (
    deepprime_is_author_train_fold,
    oped_is_author_train_fold,
    optiprime_is_author_train_fold,
)


def test_deepprime_nan_and_test_token():
    assert deepprime_is_author_train_fold(None) is True
    assert deepprime_is_author_train_fold(float("nan")) is True
    assert deepprime_is_author_train_fold("Test") is False
    assert deepprime_is_author_train_fold("-1") is False
    assert deepprime_is_author_train_fold(0) is True
    assert deepprime_is_author_train_fold(-1) is False
    assert deepprime_is_author_train_fold("unknown") is True


def test_optiprime_nan_is_train_unknown_string_is_train():
    assert optiprime_is_author_train_fold(None) is True
    assert optiprime_is_author_train_fold(float("nan")) is True
    assert optiprime_is_author_train_fold("test") is True
    assert optiprime_is_author_train_fold(-1) is False
    assert optiprime_is_author_train_fold(2) is True


def test_oped_nan_is_not_train():
    assert oped_is_author_train_fold(None) is False
    assert oped_is_author_train_fold(float("nan")) is False
    assert oped_is_author_train_fold("test") is False
    assert oped_is_author_train_fold(0) is True
    assert oped_is_author_train_fold(-1.0) is False

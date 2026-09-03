"""Tests for base-model eval split flags (run_x → library-diverse fold x)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from eval_split_args import (  # noqa: E402
    eval_result_cell_key,
    eval_result_cell_key_from_record,
    evaluation_split_cli_args,
    original_fold_test_value_from_args,
    split_plan,
)


def test_deepprime_clinvar_uses_author_test_fold():
    args = evaluation_split_cli_args(
        model="deepprime",
        study="deepprime",
        datasets="deepprime-clinvar",
        cv_run=None,
    )
    assert args == ["--use-original-fold"]
    assert original_fold_test_value_from_args(args) is None


def test_pridict2_library_diverse_tests_matching_run_fold():
    args = evaluation_split_cli_args(
        model="pridict2",
        study="pridict2",
        datasets=["library-diverse"],
        cv_run=3,
    )
    assert args == ["--use-original-fold", "--original-fold-test-value", "3"]
    plan = split_plan(
        model="pridict2",
        study="pridict2",
        datasets="library-diverse",
        cv_run=0,
    )
    assert plan["use_original_fold"] is True
    assert plan["original_fold_test_value"] == 0


def test_pridict2_library1_stays_random_holdout():
    args = evaluation_split_cli_args(
        model="pridict2",
        study="pridict1",
        datasets="library1",
        cv_run=2,
    )
    assert args == ["--no-use-original-fold"]


def test_pridict2_library_diverse_without_run_stays_random():
    args = evaluation_split_cli_args(
        model="pridict2",
        study="pridict2",
        datasets="library-diverse",
        cv_run=None,
    )
    assert args == ["--no-use-original-fold"]


def test_other_models_on_library_diverse_stay_random():
    args = evaluation_split_cli_args(
        model="optiprime",
        study="pridict2",
        datasets="library-diverse",
        cv_run=1,
    )
    assert args == ["--no-use-original-fold"]


def test_cell_key_includes_fold_only_when_author_fold_set():
    random_key = eval_result_cell_key(
        model="pridict2",
        weights="pridict1_1__exp_x__run_0__HEK",
        benchmark_name="pridict2-library-diverse__hek293t",
        cell_line="hek293t",
    )
    fold_key = eval_result_cell_key(
        model="pridict2",
        weights="pridict1_1__exp_x__run_0__HEK",
        benchmark_name="pridict2-library-diverse__hek293t",
        cell_line="hek293t",
        original_fold_test_value=0,
    )
    assert random_key != fold_key
    assert fold_key.endswith("|fold:0")
    old = eval_result_cell_key_from_record(
        {
            "model": "pridict2",
            "weights": "pridict1_1__exp_x__run_0__HEK",
            "benchmark_name": "pridict2-library-diverse__hek293t",
            "cell_line": "hek293t",
        }
    )
    assert old == random_key


def test_cli_json():
    import io
    from contextlib import redirect_stdout

    from eval_split_args import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(
            [
                "--json",
                "--model",
                "pridict2",
                "--study",
                "pridict2",
                "--datasets",
                "library-diverse",
                "--cv-run",
                "4",
            ]
        ) == 0
    payload = json.loads(buf.getvalue())
    assert payload["original_fold_test_value"] == 4
    assert payload["args"][-1] == "4"

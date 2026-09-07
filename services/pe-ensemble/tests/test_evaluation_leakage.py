"""Tests for train/test data-leak detection during evaluation."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from pe_ensemble.evaluation import leakage
from pe_ensemble.evaluation.leakage import (
    REASON_NO_ORIGINAL_TEST_SPLIT,
    REASON_TRAIN_TEST_OVERLAP,
    REASON_UNVERIFIABLE_PROVENANCE,
    assess_ensemble_leakage,
    assess_leakage,
    collect_ensemble_training_loci,
    dataset_names_from_training,
)
from pe_ensemble.evaluation.runner import execute_evaluation
from pe_ensemble.evaluation.schemas import EvaluationRequest
from pe_ensemble.training.data import ModelFormatFetchResult
from pe_ensemble.training.schemas import SplitQueryParams


def _test_df(uids: list[str], *, split_source: str = "group_id") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_uid": uids,
            "split": ["test"] * len(uids),
            "split_source": [split_source] * len(uids),
            "Efficiency": [1.0] * len(uids),
        }
    )


def _split(use_original_fold: bool = False) -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_2",
        train_pct=0.8,
        test_pct=0.2,
        use_original_fold=use_original_fold,
    )


def test_overlap_with_recorded_training_loci_is_a_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa", "ps:bbb"},
    )
    result = assess_leakage(
        test_df=_test_df(["ps:bbb", "ps:ccc"]),
        split=_split(),
        model="deepprime",
        weights_id="w1",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_TRAIN_TEST_OVERLAP
    assert result.detail["n_overlap_loci"] == 1
    assert "ps:bbb" in result.detail["example_overlap_target_uids"]


def test_restrict_to_author_holdout_drops_unlabeled_mix():
    mixed = pd.concat(
        [
            _test_df(["ps:aaa", "ps:bbb"], split_source="original_fold"),
            _test_df(["ps:endo"], split_source="group_id"),
        ],
        ignore_index=True,
    )
    filtered, n_dropped = leakage.restrict_to_author_holdout_rows(mixed)
    assert n_dropped == 1
    assert list(filtered["target_uid"]) == ["ps:aaa", "ps:bbb"]
    assert filtered["split_source"].eq("original_fold").all()


def test_restrict_to_author_holdout_leaves_pure_synthetic():
    synthetic = _test_df(["ps:ccc", "ps:ddd"], split_source="group_id")
    filtered, n_dropped = leakage.restrict_to_author_holdout_rows(synthetic)
    assert n_dropped == 0
    assert len(filtered) == 2


def test_author_holdout_mixed_with_unlabeled_is_not_a_split_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:train"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: {"filters": {"dataset": ["deeppe-ht", "deeppe-endo"]}},
    )
    mixed = pd.concat(
        [
            _test_df(["ps:test-a", "ps:test-b"], split_source="original_fold"),
            _test_df(["ps:endo"], split_source="group_id"),
        ],
        ignore_index=True,
    )
    filtered, _ = leakage.restrict_to_author_holdout_rows(mixed)
    result = assess_leakage(
        test_df=filtered,
        split=_split(use_original_fold=True),
        model="oped",
        weights_id="w1",
        eval_datasets=["deeppe-ht", "deeppe-type", "deeppe-position", "deeppe-endo"],
    )
    assert result is None


def test_exclude_overlapping_loci_keeps_non_overlapping_rows():
    exclusion = leakage.exclude_overlapping_loci(
        _test_df(["ps:bbb", "ps:ccc", "ps:bbb"]),
        {"ps:bbb"},
    )
    assert exclusion is not None
    assert not exclusion.is_empty
    assert exclusion.n_overlap_loci == 1
    assert list(exclusion.filtered_df["target_uid"]) == ["ps:ccc"]
    warning = exclusion.warning_payload()
    assert warning["action"] == "excluded_overlap_loci"
    assert warning["n_test_loci_after"] == 1


def test_exclude_overlapping_loci_empty_when_fully_overlapping():
    exclusion = leakage.exclude_overlapping_loci(
        _test_df(["ps:bbb", "ps:aaa"]),
        {"ps:aaa", "ps:bbb", "ps:ccc"},
    )
    assert exclusion is not None and exclusion.is_empty
    assert exclusion.n_loci_after == 0


def test_disjoint_recorded_training_loci_is_not_a_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa", "ps:bbb"},
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc", "ps:ddd"]),
        split=_split(),
        model="deepprime",
        weights_id="w1",
    )
    assert result is None


def test_unknown_provenance_synthetic_test_with_original_fold_requested(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="group_id"),
        split=_split(use_original_fold=True),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT


def test_unknown_provenance_author_holdout_is_trusted(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is None


def test_unknown_provenance_synthetic_without_original_fold_is_unverifiable(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry, "load_training_loci", lambda model, weights: None
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="group_id"),
        split=_split(use_original_fold=False),
        model="deepprime",
        weights_id="vendor",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_UNVERIFIABLE_PROVENANCE


def test_execute_evaluation_excludes_partial_overlap_and_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:bbb"},
    )
    # runner imports weights_registry directly
    monkeypatch.setattr(
        "pe_ensemble.evaluation.runner.weights_registry.load_training_loci",
        lambda model, weights: {"ps:bbb"},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="some/benchmark",
        weights="w1",
        study="some",
        dataset="benchmark",
        allow_data_leak=False,
    )
    from pe_ensemble.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:bbb", "ps:ccc"]))

    class _StubModel:
        def evaluate(self, test_df, weights):
            assert list(test_df["target_uid"]) == ["ps:ccc"]
            return {"pearson": 0.5, "n_samples": len(test_df)}

    with patch("pe_ensemble.evaluation.runner.fetch_model_format_result", return_value=fetch), patch(
        "pe_ensemble.evaluation.runner.ModelFactory.create_model", return_value=_StubModel()
    ):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["metrics"] is not None
    assert result["n_samples"] == 1
    assert result["leak_warning"]["action"] == "excluded_overlap_loci"
    assert result["leak_warning"]["n_overlap_loci"] == 1
    manifest = get_job(job_id)
    assert manifest["status"] == "succeeded"


def test_execute_evaluation_emits_parseable_leak_error_on_full_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:bbb", "ps:ccc"},
    )
    monkeypatch.setattr(
        "pe_ensemble.evaluation.runner.weights_registry.load_training_loci",
        lambda model, weights: {"ps:bbb", "ps:ccc"},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="some/benchmark",
        weights="w1",
        study="some",
        dataset="benchmark",
    )
    from pe_ensemble.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:bbb", "ps:ccc"]))

    with patch("pe_ensemble.evaluation.runner.fetch_model_format_result", return_value=fetch):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["status"] == "error"
    assert result["error_type"] == "data_leak"
    assert result["leak_reason"] == REASON_TRAIN_TEST_OVERLAP
    assert result["metrics"] is None
    assert result["leak"]["action"] == "excluded_overlap_loci"
    assert result["leak"]["n_test_loci_after"] == 0

    manifest = get_job(job_id)
    assert manifest["status"] == "failed"
    assert manifest["error"].startswith("data_leak:")
    assert manifest["result"]["error_type"] == "data_leak"


def test_execute_evaluation_allow_data_leak_override(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:bbb"},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="some/benchmark",
        weights="w1",
        study="some",
        dataset="benchmark",
        allow_data_leak=True,
    )
    from pe_ensemble.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:bbb", "ps:ccc"]))

    class _StubModel:
        def evaluate(self, test_df, weights):
            return {"pearson": 0.5, "n_samples": len(test_df)}

    with patch("pe_ensemble.evaluation.runner.fetch_model_format_result", return_value=fetch), patch(
        "pe_ensemble.evaluation.runner.ModelFactory.create_model", return_value=_StubModel()
    ):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["metrics"] is not None
    assert result["n_samples"] == 2
    assert result["leak_warning"]["reason"] == REASON_TRAIN_TEST_OVERLAP
    assert "action" not in result["leak_warning"]
    manifest = get_job(job_id)
    assert manifest["status"] == "succeeded"


def test_collect_ensemble_training_loci_unions_member_provenance(monkeypatch):
    def _load(model, weights):
        mapping = {
            ("pridict2", "a"): {"ps:aaa", "ps:bbb"},
            ("pridict2", "b"): {"ps:bbb", "ps:ccc"},
        }
        return mapping.get((model, weights))

    monkeypatch.setattr(leakage.weights_registry, "load_training_loci", _load)
    loci, detail = collect_ensemble_training_loci(
        [
            {"model_name": "pridict2", "weights": "a"},
            {"model_name": "pridict2", "weights": "b"},
        ]
    )
    assert loci == {"ps:aaa", "ps:bbb", "ps:ccc"}
    assert detail["all_members_have_provenance"] is True
    assert detail["n_target_loci"] == 3
    assert detail["loci_fingerprint"]


def test_collect_ensemble_training_loci_none_when_any_member_missing(monkeypatch):
    def _load(model, weights):
        if weights == "a":
            return {"ps:aaa"}
        return None

    monkeypatch.setattr(leakage.weights_registry, "load_training_loci", _load)
    loci, detail = collect_ensemble_training_loci(
        [
            {"model_name": "pridict2", "weights": "a"},
            {"model_name": "pridict2", "weights": "b"},
        ]
    )
    assert loci is None
    assert detail["all_members_have_provenance"] is False
    assert detail["n_members_missing_provenance"] == 1


def test_assess_ensemble_leakage_uses_unioned_loci(monkeypatch):
    def _load(model, weights):
        mapping = {
            ("pridict2", "a"): {"ps:aaa"},
            ("pridict2", "b"): {"ps:bbb"},
        }
        return mapping.get((model, weights))

    monkeypatch.setattr(leakage.weights_registry, "load_training_loci", _load)
    members = [
        {"model_name": "pridict2", "weights": "a"},
        {"model_name": "pridict2", "weights": "b"},
    ]
    result = assess_ensemble_leakage(
        test_df=_test_df(["ps:bbb", "ps:ccc"]),
        split=_split(),
        members=members,
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_TRAIN_TEST_OVERLAP
    assert result.detail["n_overlap_loci"] == 1
    assert result.detail["ensemble_training_loci"]["n_target_loci"] == 2


def test_dataset_names_from_training_normalizes_underscores():
    names = dataset_names_from_training(
        {
            "filters": {"dataset": ["library1", "library_diverse"]},
            "data_provenance": {
                "vendor_training_lineage": [{"dataset": "lib_cv", "study": "optiprime"}]
            },
        }
    )
    assert names == {"library1", "library-diverse", "lib-cv"}


def test_synthetic_library1_eval_aborts_even_without_uid_overlap(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: {"filters": {"dataset": ["library1"]}},
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc", "ps:ddd"]),
        split=_split(),
        model="pridict2",
        weights_id="w1",
        eval_datasets="library1",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result.detail["in_domain_datasets"] == ["library1"]
    assert result.detail["test_is_author_holdout"] is False


def test_weight_respects_author_holdout_defaults_true_when_key_missing():
    assert leakage.weight_respects_author_holdout(None) is True
    assert leakage.weight_respects_author_holdout({"filters": {"dataset": ["library-diverse"]}}) is True
    assert (
        leakage.weight_respects_author_holdout(
            {"data_provenance": {"loci_recorded": True}}
        )
        is True
    )
    assert (
        leakage.weight_respects_author_holdout(
            {"data_provenance": {"has_original_test_split": True}}
        )
        is True
    )
    assert (
        leakage.weight_respects_author_holdout(
            {"data_provenance": {"has_original_test_split": False}}
        )
        is False
    )


def _optiprime_training_meta() -> dict:
    return {
        "filters": {
            "dataset": [
                "lib-mmr",
                "lib-cv",
                "library1",
                "deepprime-clinvar",
                "library-diverse",
            ]
        },
        "data_provenance": {"has_original_test_split": False},
    }


def test_optiprime_library_diverse_author_holdout_aborts(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: _optiprime_training_meta(),
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc", "ps:ddd"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        model="optiprime",
        weights_id="base",
        eval_datasets="library-diverse",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result.detail["in_domain_datasets"] == ["library-diverse"]
    assert result.detail["test_is_author_holdout"] is True
    assert result.detail["weight_respects_author_holdout"] is False


def test_optiprime_clinvar_author_holdout_aborts(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: _optiprime_training_meta(),
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        model="optiprime",
        weights_id="base",
        eval_datasets="deepprime-clinvar",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result.detail["in_domain_datasets"] == ["deepprime-clinvar"]
    assert result.detail["test_is_author_holdout"] is True
    assert result.detail["weight_respects_author_holdout"] is False


def test_author_holdout_of_training_dataset_is_not_a_split_leak(monkeypatch):
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: {
            "filters": {"dataset": ["library-diverse"]},
            "data_provenance": {"has_original_test_split": True},
        },
    )
    result = assess_leakage(
        test_df=_test_df(["ps:ccc"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        model="pridict2",
        weights_id="w1",
        eval_datasets="library-diverse",
    )
    assert result is None


def test_execute_evaluation_aborts_optiprime_library_diverse_holdout(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: _optiprime_training_meta(),
    )

    request = EvaluationRequest(
        model_name="optiprime",
        benchmark_name="pridict2-library-diverse__hek293t",
        weights="base",
        study="pridict2",
        dataset="library-diverse",
        auto_training_benchmark=False,
    )
    from pe_ensemble.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(
        df=_test_df(["ps:ccc", "ps:ddd"], split_source="original_fold")
    )

    with patch("pe_ensemble.evaluation.runner.fetch_model_format_result", return_value=fetch), patch(
        "pe_ensemble.evaluation.runner.ModelFactory.create_model"
    ) as create_model:
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    create_model.assert_not_called()
    assert result["status"] == "error"
    assert result["error_type"] == "data_leak"
    assert result["leak_reason"] == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result["metrics"] is None
    assert result["leak"]["in_domain_datasets"] == ["library-diverse"]
    assert "excluded_overlap_loci" not in str(result.get("leak", {}))
    manifest = get_job(job_id)
    assert manifest["status"] == "failed"


def test_execute_evaluation_aborts_synthetic_library1(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_loci",
        lambda model, weights: {"ps:aaa"},
    )
    monkeypatch.setattr(
        leakage.weights_registry,
        "load_training_metadata",
        lambda model, weights: {"filters": {"dataset": ["library1"]}},
    )

    request = EvaluationRequest(
        model_name="deepprime",
        benchmark_name="pridict1-library1",
        weights="w1",
        study="pridict1",
        dataset="library1",
    )
    from pe_ensemble.evaluation.jobs import create_job, get_job

    job_id = create_job(request)
    fetch = ModelFormatFetchResult(df=_test_df(["ps:ccc", "ps:ddd"]))

    with patch("pe_ensemble.evaluation.runner.fetch_model_format_result", return_value=fetch):
        result = execute_evaluation(request, job_id=job_id, device_id="cpu")

    assert result["status"] == "error"
    assert result["error_type"] == "data_leak"
    assert result["leak_reason"] == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result["metrics"] is None
    assert result["leak"]["in_domain_datasets"] == ["library1"]
    manifest = get_job(job_id)
    assert manifest["status"] == "failed"


def test_assess_ensemble_leakage_aborts_synthetic_library1(monkeypatch):
    def _load_loci(model, weights):
        return {"ps:aaa"}

    def _load_meta(model, weights):
        return {"filters": {"dataset": ["library1", "library-diverse"]}}

    monkeypatch.setattr(leakage.weights_registry, "load_training_loci", _load_loci)
    monkeypatch.setattr(leakage.weights_registry, "load_training_metadata", _load_meta)
    result = assess_ensemble_leakage(
        test_df=_test_df(["ps:zzz"]),
        split=_split(),
        members=[
            {"model_name": "pridict2", "weights": "a"},
            {"model_name": "pridict2", "weights": "b"},
        ],
        eval_datasets="library1",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result.detail["in_domain_datasets"] == ["library1"]


def test_assess_ensemble_leakage_aborts_when_any_member_lacks_author_holdout(monkeypatch):
    def _load_loci(model, weights):
        return {"ps:aaa"}

    def _load_meta(model, weights):
        if model == "optiprime":
            return _optiprime_training_meta()
        return {
            "filters": {"dataset": ["library-diverse"]},
            "data_provenance": {"has_original_test_split": True},
        }

    monkeypatch.setattr(leakage.weights_registry, "load_training_loci", _load_loci)
    monkeypatch.setattr(leakage.weights_registry, "load_training_metadata", _load_meta)
    result = assess_ensemble_leakage(
        test_df=_test_df(["ps:zzz"], split_source="original_fold"),
        split=_split(use_original_fold=True),
        members=[
            {"model_name": "pridict2", "weights": "a"},
            {"model_name": "optiprime", "weights": "base"},
        ],
        eval_datasets="library-diverse",
    )
    assert result is not None and result.is_leak
    assert result.reason == REASON_NO_ORIGINAL_TEST_SPLIT
    assert result.detail["weight_respects_author_holdout"] is False

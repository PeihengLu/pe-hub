"""Tests for eval JSON extraction, OptiPrime log repair, and PRIDICT2 CV grouping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from summarize_eval_results import (  # noqa: E402
    aggregate_cv,
    extract_json_object,
    flatten_row,
    pridict2_head_from_weights,
    repair_cli_failures_from_logs,
    repair_ensemble_payloads_from_logs,
)


def test_extract_json_skips_optiprime_syn_brace():
    stdout = (
        "INFO:root:  syn{50}\n"
        "INFO:app.evaluation.runner:Evaluation succeeded; n_samples=14\n"
        "{\n"
        '  "model": "optiprime",\n'
        '  "benchmark_name": "deeppe-pooled__hct116",\n'
        '  "weights": "base",\n'
        '  "n_samples": 14,\n'
        '  "metrics": {"pearson": 0.49, "spearman": 0.67, "n_samples": 14},\n'
        '  "leak_warning": {"reason": "train_test_overlap", "n_overlap_loci": 1}\n'
        "}\n"
    )
    payload = extract_json_object(stdout)
    assert payload is not None
    assert payload["model"] == "optiprime"
    assert payload["metrics"]["pearson"] == 0.49
    assert payload["n_samples"] == 14


def test_extract_json_empty():
    assert extract_json_object("") is None
    assert extract_json_object("syn{50} only") is None


def test_extract_json_prefers_combined_ensemble_over_member_metrics():
    stdout = json.dumps(
        {
            "ensemble_name": "pridict2-ensemble-HEK-run0",
            "combine": "mean",
            "n_samples": 4445,
            "metrics": {"pearson": 0.866, "spearman": 0.848, "n_samples": 4445},
            "member_metrics": [
                {
                    "model_name": "pridict2",
                    "weights": "pridict1_1__run_0__HEK",
                    "metrics": {"pearson": 0.846, "spearman": 0.821, "n_samples": 4445},
                },
                {
                    "model_name": "pridict2",
                    "weights": "pridict1_2__run_0__HEK",
                    "metrics": {"pearson": 0.864, "spearman": 0.857, "n_samples": 4445},
                },
            ],
        },
        indent=2,
    )
    payload = extract_json_object("INFO:app.ensemble.runner:starting\n" + stdout)
    assert payload is not None
    assert payload["ensemble_name"] == "pridict2-ensemble-HEK-run0"
    assert payload["metrics"]["pearson"] == 0.866
    assert payload["n_samples"] == 4445
    assert len(payload["member_metrics"]) == 2


def test_repair_ensemble_payloads_replaces_member_fragment(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    stdout = json.dumps(
        {
            "ensemble_name": "pridict2-ensemble-HEK-run0",
            "combine": "mean",
            "n_samples": 4445,
            "metrics": {"pearson": 0.866, "spearman": 0.848, "n_samples": 4445},
            "member_metrics": [
                {
                    "model_name": "pridict2",
                    "weights": "pridict1_2__run_0__HEK",
                    "metrics": {"pearson": 0.864, "spearman": 0.857, "n_samples": 4445},
                }
            ],
        }
    )
    (
        log_dir
        / "pridict2__ensemble__run_0__HEK__pridict2-library-diverse__hek293t__fold_0.stdout"
    ).write_text(stdout)
    records = [
        {
            "model": "pridict2",
            "weights": "ensemble__run_0__HEK",
            "benchmark_name": "pridict2-library-diverse__hek293t",
            "original_fold_test_value": 0,
            "ensemble": True,
            "n_samples": None,
            "metrics": {"pearson": 0.864, "spearman": 0.857, "n_samples": 4445},
        }
    ]
    assert repair_ensemble_payloads_from_logs(records, log_dir) == 1
    assert records[0]["metrics"]["pearson"] == 0.866
    assert records[0]["n_samples"] == 4445
    assert records[0]["weights"] == "ensemble__run_0__HEK"


def test_pridict2_head_from_weights():
    assert pridict2_head_from_weights("exp__run_0__HEK") == "HEK"
    assert pridict2_head_from_weights("exp__run_0__K562") == "K562"
    assert pridict2_head_from_weights("exp__run_0__K562MLH1dn") == "K562MLH1dn"
    assert pridict2_head_from_weights("DeepPrime_base") is None


def test_cv_aggregate_keeps_heads_separate():
    rows = []
    for head, pearson in (("HEK", 0.8), ("K562", 0.4)):
        for run in range(5):
            rows.append(
                flatten_row(
                    {
                        "model": "pridict2",
                        "weights": f"exp__run_{run}__{head}",
                        "experiment_id": "exp",
                        "cv_run": run,
                        "benchmark_name": "library1",
                        "study": "pridict1",
                        "datasets": ["library1"],
                        "cell_line": "hek293t",
                        "status": "ok",
                        "n_samples": 10,
                        "metrics": {"pearson": pearson, "spearman": pearson},
                    }
                )
            )
    agg = {row["pridict2_head"]: row for row in aggregate_cv(rows)}
    assert set(agg) == {"HEK", "K562"}
    assert agg["HEK"]["n_folds"] == 5
    assert agg["K562"]["n_folds"] == 5
    assert agg["HEK"]["pearson_mean"] == 0.8
    assert agg["K562"]["pearson_mean"] == 0.4


def test_repair_cli_failure_from_optiprime_stdout(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    stdout = (
        "INFO:root:  syn{50}\n"
        "{\n"
        '  "model": "optiprime",\n'
        '  "benchmark_name": "deepprime-clinvar",\n'
        '  "weights": "base",\n'
        '  "n_samples": 28084,\n'
        '  "metrics": {"pearson": 0.26, "spearman": 0.47, "n_samples": 28084}\n'
        "}\n"
    )
    (log_dir / "optiprime__base__deepprime-clinvar.stdout").write_text(stdout)
    records = [
        {
            "model": "optiprime",
            "weights": "base",
            "benchmark_name": "deepprime-clinvar",
            "status": "error",
            "error_type": "cli_failure",
            "metrics": None,
        }
    ]
    assert repair_cli_failures_from_logs(records, log_dir) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["metrics"]["pearson"] == 0.26
    assert "error_type" not in records[0]

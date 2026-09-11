"""Tests for eval JSON extraction, OptiPrime log repair, and PRIDICT2 CV grouping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from summarize_eval_results import (  # noqa: E402
    aggregate_cv,
    comparison_table,
    extract_json_object,
    flatten_row,
    pridict2_head_from_weights,
    repair_cli_failures_from_logs,
    repair_ensemble_payloads_from_logs,
)
from paper_reported_metrics import annotate_row_with_paper  # noqa: E402


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


def test_optiprime_lib_mmr_leak_is_author_fill():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "optiprime-lib-mmr__hek293t__pe2",
                "study": "optiprime",
                "datasets": ["lib-mmr"],
                "cell_line": "hek293t",
                "pe_system": "pe2",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 3658,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["plot_hatch"] == "///"
    assert row["pearson_plot"] == 0.723
    assert row["pearson_measured"] is None
    assert row["status"] == "error"
    assert row["leak_reason"] == "no_original_test_split"


def test_optiprime_lib_mmr_pe4_leak_is_author_fill():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "optiprime-lib-mmr__hek293t__pe4",
                "study": "optiprime",
                "datasets": ["lib-mmr"],
                "cell_line": "hek293t",
                "pe_system": "pe4",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 3658,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["plot_hatch"] == "///"
    assert row["pearson_plot"] == 0.723
    assert row["paper_id"] == "optiprime-lib-mmr-hek-pe4"


def test_optiprime_lib_mmr_hela_pe2_uses_fig4b():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "optiprime-lib-mmr__hela__pe2",
                "study": "optiprime",
                "datasets": ["lib-mmr"],
                "cell_line": "hela",
                "pe_system": "pe2",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 3766,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["pearson_plot"] == 0.760
    assert row["spearman_plot"] == 0.798
    assert row["paper_id"] == "optiprime-lib-mmr-hela"


def test_optiprime_lib_mmr_hela_pe4_uses_fig4c():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "optiprime-lib-mmr__hela__pe4",
                "study": "optiprime",
                "datasets": ["lib-mmr"],
                "cell_line": "hela",
                "pe_system": "pe4",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 3766,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["pearson_plot"] == 0.803
    assert row["spearman_plot"] == 0.810
    assert row["paper_id"] == "optiprime-lib-mmr-hela-pe4"


def test_optiprime_library_diverse_leak_is_not_filled_with_hsu_number():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "pridict2-library-diverse__hek293t",
                "study": "pridict2",
                "datasets": ["library-diverse"],
                "cell_line": "hek293t",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 4530,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "leak_unfilled"
    assert row["pearson_plot"] is None
    assert row["paper_id"] is None


def test_optiprime_clinvar_leak_is_not_filled_with_hsu_number():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "optiprime",
                "weights": "base",
                "benchmark_name": "deepprime-clinvar",
                "study": "deepprime",
                "datasets": ["deepprime-clinvar"],
                "cell_line": "hek293t",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 28084,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "leak_unfilled"
    assert row["pearson_plot"] is None
    assert row["paper_id"] is None


def test_pridict2_hek_library1_leak_is_author_fill():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "pridict2",
                "weights": "ensemble__run_0__HEK",
                "benchmark_name": "pridict1-library1",
                "study": "pridict1",
                "datasets": ["library1"],
                "cell_line": "hek293t",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 18399,
                "metrics": None,
                "ensemble": True,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["plot_hatch"] == "///"
    assert row["pearson_plot"] == 0.86
    assert row["paper_id"] == "pridict2-library1-cv"


def test_pridict2_k562_library1_leak_is_not_filled():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "pridict2",
                "weights": "ensemble__run_0__K562",
                "benchmark_name": "pridict1-library1",
                "study": "pridict1",
                "datasets": ["library1"],
                "cell_line": "hek293t",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 18399,
                "metrics": None,
                "ensemble": True,
            }
        )
    )
    assert row["value_source"] == "leak_unfilled"
    assert row["pearson_plot"] is None


def test_oped_deeppe_hek_keeps_measured():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "oped",
                "weights": "base",
                "benchmark_name": "deeppe-pooled__hek293t",
                "study": "deeppe",
                "datasets": ["pooled"],
                "cell_line": "hek293t",
                "status": "ok",
                "n_samples": 5060,
                "metrics": {"pearson": 0.722, "spearman": 0.768},
            }
        )
    )
    assert row["value_source"] == "measured"
    assert abs(row["pearson_plot"] - 0.722) < 1e-9
    assert row["paper_pearson"] == 0.769
    assert row["paper_id"] == "oped-deeppe-ht-test"


def test_oped_deeppe_ht_test_only_is_close_match():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "oped",
                "weights": "base",
                "benchmark_name": "deeppe-ht-test",
                "study": "deeppe",
                "datasets": ["deeppe-ht"],
                "cell_line": "hek293t",
                "status": "ok",
                "n_samples": 4457,
                "metrics": {"pearson": 0.769, "spearman": 0.798},
            }
        )
    )
    assert row["paper_id"] == "oped-deeppe-ht-test-only"
    assert row["paper_protocol_match"] == "close"
    assert row["paper_pearson"] == 0.769


def test_oped_deeppe_hct_leak_is_author_fill():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "oped",
                "weights": "base",
                "benchmark_name": "deeppe-pooled__hct116",
                "study": "deeppe",
                "datasets": ["pooled"],
                "cell_line": "hct116",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 15,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["pearson_plot"] == 0.590
    assert row["paper_id"] == "oped-deeppe-hct"


def test_oped_deeppe_mda_leak_is_author_fill():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "oped",
                "weights": "base",
                "benchmark_name": "deeppe-pooled__mda_mb_231",
                "study": "deeppe",
                "datasets": ["pooled"],
                "cell_line": "mda_mb_231",
                "status": "error",
                "error_type": "data_leak",
                "leak_reason": "no_original_test_split",
                "n_samples": 15,
                "metrics": None,
            }
        )
    )
    assert row["value_source"] == "author_fill"
    assert row["pearson_plot"] == 0.650
    assert row["paper_id"] == "oped-deeppe-mda"


def test_deepprime_clinvar_keeps_measured_and_joins_paper():
    row = annotate_row_with_paper(
        flatten_row(
            {
                "model": "deepprime",
                "weights": "DeepPrime_base",
                "benchmark_name": "deepprime-clinvar",
                "study": "deepprime",
                "datasets": ["deepprime-clinvar"],
                "cell_line": "hek293t",
                "status": "ok",
                "n_samples": 28221,
                "metrics": {"pearson": 0.827, "spearman": 0.854},
            }
        )
    )
    assert row["value_source"] == "measured"
    assert row["plot_hatch"] == ""
    assert row["paper_pearson"] == 0.84
    assert abs(row["pearson_plot"] - 0.827) < 1e-9


def test_comparison_table_uses_cv_mean_for_pridict2():
    rows = []
    for run in range(5):
        rows.append(
            annotate_row_with_paper(
                flatten_row(
                    {
                        "model": "pridict2",
                        "weights": f"ensemble__run_{run}__HEK",
                        "experiment_id": "pridict2_ensemble",
                        "cv_run": run,
                        "benchmark_name": "pridict2-library-diverse__hek293t",
                        "study": "pridict2",
                        "datasets": ["library-diverse"],
                        "cell_line": "hek293t",
                        "status": "ok",
                        "n_samples": 4445,
                        "metrics": {"pearson": 0.87, "spearman": 0.87},
                        "ensemble": True,
                    }
                )
            )
        )
    table = comparison_table(rows, aggregate_cv(rows))
    assert len(table) == 1
    assert table[0]["paper_pearson"] == 0.90
    assert table[0]["value_source"] == "measured"
    assert abs(table[0]["pearson_delta"] - (0.87 - 0.90)) < 1e-9


def test_bar_fill_kind_distinguishes_measured_author_and_missing():
    from plot_base_model_eval import (
        BENCH_ORDER,
        FILL_AUTHOR,
        FILL_MEASURED,
        FILL_MISSING,
        HEATMAP_PANELS,
        cell_fill_kind,
    )

    assert cell_fill_kind(None) == FILL_MISSING
    assert (
        cell_fill_kind({"plot_marker": "measured", "value_source": "measured", "pearson_plot": 0.8})
        == FILL_MEASURED
    )
    assert (
        cell_fill_kind({"plot_marker": "author_fill", "value_source": "author_fill", "pearson_plot": 0.723})
        == FILL_AUTHOR
    )
    assert (
        cell_fill_kind({"plot_marker": "", "value_source": "leak_unfilled", "pearson_plot": ""})
        == FILL_MISSING
    )
    bench_keys = [key for key, _label in BENCH_ORDER]
    panel_keys = [key for panel in HEATMAP_PANELS for key, _label, _study in panel]
    assert "minsepie-insert-pooled__rc__pe2" not in bench_keys
    assert "minsepie-insert-pooled__rc__pe2" not in panel_keys
    assert "minsepie-insert-pooled__hek293t__pe2" in bench_keys
    assert "deeppe-pooled__hct116" not in panel_keys
    assert "deeppe-pooled__mda_mb_231" not in panel_keys
    first_row = [key for key, _label, _study in HEATMAP_PANELS[0]]
    assert "minsepie-insert-pooled__hek293t__pe2" in first_row


#!/usr/bin/env python3
"""Flatten peen evaluate JSONL results into summary.csv (+ CV mean/std).

Usage:
  python scripts/experiments/summarize_eval_results.py results/base_model_eval/<RUN_ID>/results.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


SUMMARY_COLUMNS = [
    "model",
    "weights",
    "experiment_id",
    "cv_run",
    "pridict2_head",
    "benchmark_name",
    "study",
    "datasets",
    "cell_line",
    "original_fold_test_value",
    "status",
    "n_samples",
    "pearson",
    "spearman",
    "leak_reason",
    "skip_reason",
    "error_type",
    "device",
]

AGG_COLUMNS = [
    "model",
    "experiment_id",
    "pridict2_head",
    "benchmark_name",
    "study",
    "datasets",
    "cell_line",
    "n_folds",
    "n_ok",
    "pearson_mean",
    "pearson_std",
    "spearman_mean",
    "spearman_std",
    "n_samples_mean",
]

# Longest first so ``__K562MLH1dn`` is not parsed as ``__K562``.
_PRIDICT2_HEAD_SUFFIXES = ("K562MLH1dn", "K562", "HEK")


def _metric(metrics: Any, key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    # Common shapes: {"pearson": x, "spearman": y} or nested under "metrics"
    value = metrics.get(key)
    if value is None:
        # DeepPrime-style aliases
        aliases = {
            "pearson": ("pearson_r", "pearsonr", "r"),
            "spearman": ("spearman_r", "spearmanr", "rho"),
        }
        for alt in aliases.get(key, ()):
            if metrics.get(alt) is not None:
                value = metrics[alt]
                break
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _datasets_str(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def pridict2_head_from_weights(weights: Any) -> Optional[str]:
    text = str(weights or "")
    for head in _PRIDICT2_HEAD_SUFFIXES:
        if text.endswith(f"__{head}"):
            return head
    return None


def _looks_like_eval_payload(obj: dict[str, Any]) -> bool:
    return (
        "metrics" in obj
        or bool(obj.get("error_type"))
        or bool(obj.get("skipped"))
        or ("model" in obj and "benchmark_name" in obj)
    )


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse the peen evaluation JSON object from mixed stdout.

    Vendor logs may contain ``{`` earlier (OptiPrime logs ``syn{50}``) and the
    result object itself is nested, so neither the first nor the last brace is
    always the payload. Scan backwards and keep the first object that looks
    like an evaluate record.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    idx = text.rfind("{")
    fallback: Optional[dict[str, Any]] = None
    while idx >= 0:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.rfind("{", 0, idx)
            continue
        if isinstance(obj, dict):
            if _looks_like_eval_payload(obj):
                return obj
            fallback = obj
        idx = text.rfind("{", 0, idx)
    return fallback


def stdout_log_path(
    log_dir: Path,
    model: Any,
    weights: Any,
    benchmark_name: Any,
    original_fold_test_value: Any = None,
) -> Path:
    fold = ""
    if original_fold_test_value is not None and original_fold_test_value != "":
        fold = f"__fold_{int(float(original_fold_test_value))}"
    safe = f"{model}__{weights}__{benchmark_name}{fold}".replace("/", "_").replace(":", "_")
    return log_dir / f"{safe}.stdout"


def repair_cli_failures_from_logs(
    records: list[dict[str, Any]], log_dir: Path
) -> int:
    """Re-parse stdout for ``cli_failure`` rows whose logs actually succeeded."""
    repaired = 0
    for record in records:
        if record.get("error_type") != "cli_failure":
            continue
        path = stdout_log_path(
            log_dir,
            record.get("model"),
            record.get("weights"),
            record.get("benchmark_name"),
            record.get("original_fold_test_value"),
        )
        if not path.is_file():
            continue
        payload = extract_json_object(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if not isinstance(payload, dict) or payload.get("metrics") is None:
            continue
        record.update(payload)
        record["status"] = "ok"
        record.pop("error_type", None)
        record.pop("stderr_tail", None)
        repaired += 1
    return repaired


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def flatten_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    status = record.get("status")
    if status is None:
        if record.get("skipped"):
            status = "skipped"
        elif record.get("error_type"):
            status = "error"
        elif metrics is not None:
            status = "ok"
        else:
            status = "unknown"
    return {
        "model": record.get("model"),
        "weights": record.get("weights"),
        "experiment_id": record.get("experiment_id") or record.get("weights"),
        "cv_run": record.get("cv_run"),
        "pridict2_head": pridict2_head_from_weights(record.get("weights")),
        "benchmark_name": record.get("benchmark_name"),
        "study": record.get("study"),
        "datasets": _datasets_str(record.get("datasets")),
        "cell_line": record.get("cell_line"),
        "original_fold_test_value": record.get("original_fold_test_value"),
        "status": status,
        "n_samples": record.get("n_samples"),
        "pearson": _metric(metrics, "pearson"),
        "spearman": _metric(metrics, "spearman"),
        "leak_reason": record.get("leak_reason"),
        "skip_reason": record.get("skip_reason"),
        "error_type": record.get("error_type"),
        "device": record.get("device"),
    }


def aggregate_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std over CV folds for experiments that have multiple cv_run values."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("cv_run") is None:
            continue
        if row.get("status") != "ok":
            # still group so n_ok can be < n_folds
            pass
        key = (
            row.get("model"),
            row.get("experiment_id"),
            row.get("pridict2_head"),
            row.get("benchmark_name"),
            row.get("study"),
            row.get("datasets"),
            row.get("cell_line"),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        model, experiment_id, pridict2_head, benchmark_name, study, datasets, cell_line = key
        ok = [m for m in members if m.get("status") == "ok" and m.get("pearson") is not None]
        pearsons = [float(m["pearson"]) for m in ok]
        spearmans = [float(m["spearman"]) for m in ok if m.get("spearman") is not None]
        n_samples = [float(m["n_samples"]) for m in ok if m.get("n_samples") is not None]

        def _mean(vals: list[float]) -> Optional[float]:
            return statistics.fmean(vals) if vals else None

        def _std(vals: list[float]) -> Optional[float]:
            return statistics.stdev(vals) if len(vals) >= 2 else (0.0 if len(vals) == 1 else None)

        out.append(
            {
                "model": model,
                "experiment_id": experiment_id,
                "pridict2_head": pridict2_head,
                "benchmark_name": benchmark_name,
                "study": study,
                "datasets": datasets,
                "cell_line": cell_line,
                "n_folds": len(members),
                "n_ok": len(ok),
                "pearson_mean": _mean(pearsons),
                "pearson_std": _std(pearsons),
                "spearman_mean": _mean(spearmans),
                "spearman_std": _std(spearmans),
                "n_samples_mean": _mean(n_samples),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def print_console_summary(rows: list[dict[str, Any]], agg: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status") or "unknown")] += 1
    print(f"rows={len(rows)}  status_counts={dict(sorted(counts.items()))}")
    leak = sum(1 for r in rows if r.get("error_type") == "data_leak")
    if leak:
        print(f"data_leak aborts: {leak}")
    if agg:
        print(f"cv aggregates (experiment × head × benchmark): {len(agg)}")
        for row in agg[:12]:
            head = row.get("pridict2_head")
            head_bit = f"/{head}" if head else ""
            print(
                f"  {row['model']}/{row['experiment_id']}{head_bit} @ {row['benchmark_name']}: "
                f"pearson={row['pearson_mean']!s}±{row['pearson_std']!s} "
                f"(n_ok={row['n_ok']}/{row['n_folds']})"
            )
        if len(agg) > 12:
            print(f"  ... {len(agg) - 12} more")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path, help="Path to results.jsonl")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV (default: <jsonl-dir>/summary.csv)",
    )
    parser.add_argument(
        "--agg-csv",
        type=Path,
        default=None,
        help="CV mean/std CSV (default: <jsonl-dir>/summary_cv_mean_std.csv)",
    )
    parser.add_argument(
        "--repair-from-logs",
        action="store_true",
        help=(
            "Re-parse sibling logs/*.stdout for cli_failure rows whose vendor "
            "logs contain a successful peen JSON object (OptiPrime syn{50})."
        ),
    )
    args = parser.parse_args(argv)

    jsonl_path = args.results_jsonl.resolve()
    if not jsonl_path.is_file():
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        return 1

    out_dir = jsonl_path.parent
    summary_path = args.summary_csv or (out_dir / "summary.csv")
    agg_path = args.agg_csv or (out_dir / "summary_cv_mean_std.csv")

    records = load_records(jsonl_path)
    if args.repair_from_logs:
        repaired = repair_cli_failures_from_logs(records, out_dir / "logs")
        if repaired:
            jsonl_path.write_text(
                "".join(json.dumps(record, default=str) + "\n" for record in records),
                encoding="utf-8",
            )
            print(f"Repaired {repaired} cli_failure row(s) from logs")
    rows = [flatten_row(record) for record in records]
    write_csv(summary_path, rows, SUMMARY_COLUMNS)

    agg = aggregate_cv(rows)
    write_csv(agg_path, agg, AGG_COLUMNS)

    print(f"Wrote {summary_path} ({len(rows)} rows)")
    print(f"Wrote {agg_path} ({len(agg)} rows)")
    print_console_summary(rows, agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

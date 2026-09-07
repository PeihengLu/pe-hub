#!/usr/bin/env bash
# Stage 03 — Evaluate leftover seeds (if 01 skipped eval) and flatten summaries.
#
# Default: resume peen evaluate via datasheet-benchmark (--skip-existing).
# Per-seed 01 jobs already evaluate; this is then a cheap aggregation pass.
#
# Then flatten all cells under results/<RUN_ID>/ into matrix summary.csv.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh
#   RUN_ID=20260901T140000 ./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh
#
# ARC (single job — do not fan out per cell):
#   RUN_ID=<id> ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    export MODELS="${MODEL}"
    export BENCHMARKS="${BENCHMARK}"
fi

if [[ -z "${RUN_ID:-}" && -f "${RESULTS_DIR}/LATEST_RUN_ID" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "${RESULTS_DIR}/LATEST_RUN_ID")"
fi
ensure_run_id

print_benchmark_banner "03 Evaluate matrix (datasheet-benchmark test holdout)"
echo "RUN_ID:  ${RUN_ID}"
echo ""

export SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

while IFS= read -r row; do
    parse_matrix_row "${row}"
    while read -r model; do
        [[ -n "${model}" ]] || continue
        key="$(cell_key "${model}" "${MATRIX_BENCH}")"
        export FIXED_HP_JSON="$(fixed_tune_hp_json "${model}")"

        echo "======================================"
        echo "EVAL ${model} @ ${MATRIX_BENCH}"
        echo "======================================"

        out_dir="$(cell_out_dir "${model}" "${MATRIX_BENCH}")"
        if [[ ! -d "${out_dir}/state" ]]; then
            echo "Warning: no 01/02 output at ${out_dir}/state — skip eval for ${key}" >&2
            continue
        fi

        # Re-run with skip-existing: 01 already evals; this resumes leftover seeds.
        if ! run_datasheet_benchmark_cell "${model}" 0; then
            echo "Warning: evaluate/resume failed for ${key}" >&2
        fi
        echo ""
    done < <(selected_models | tr ' ' '\n')
done < <(selected_matrix_rows)

OUT_DIR="${RESULTS_DIR}/${RUN_ID}"
RESULTS_JSONL="${OUT_DIR}/results.jsonl"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
AGG_CSV="${OUT_DIR}/summary_mean_std.csv"

python - "${OUT_DIR}" "${RESULTS_JSONL}" "${SUMMARY_CSV}" "${AGG_CSV}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out_dir = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
agg_path = Path(sys.argv[4])

rows = []
for path in sorted(out_dir.glob("*/results.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

jsonl_path.parent.mkdir(parents=True, exist_ok=True)
with jsonl_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

fields = [
    "run_id", "model", "protocol", "repeat_id", "seed", "n_trials",
    "dataset_name", "weights_id", "status", "n_samples",
    "test_spearman", "test_pearson", "test_mse", "best_value",
]
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

groups = defaultdict(list)
for row in rows:
    if row.get("status") != "ok":
        continue
    key = (row.get("model"), row.get("dataset_name") or row.get("protocol"))
    if row.get("test_spearman") is not None:
        groups[key].append(row)

agg_fields = [
    "model", "dataset_name", "n_ok",
    "test_spearman_mean", "test_spearman_std",
    "test_pearson_mean", "test_pearson_std",
]
with agg_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=agg_fields)
    writer.writeheader()
    for (model, dataset_name), items in sorted(groups.items()):
        spears = [float(r["test_spearman"]) for r in items if r.get("test_spearman") is not None]
        pears = [float(r["test_pearson"]) for r in items if r.get("test_pearson") is not None]
        def _mean_std(values):
            if not values:
                return None, None
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean, std
        s_mean, s_std = _mean_std(spears)
        p_mean, p_std = _mean_std(pears)
        writer.writerow({
            "model": model,
            "dataset_name": dataset_name,
            "n_ok": len(items),
            "test_spearman_mean": s_mean,
            "test_spearman_std": s_std,
            "test_pearson_mean": p_mean,
            "test_pearson_std": p_std,
        })

print(f"Wrote {jsonl_path} ({len(rows)} rows)")
print(f"Wrote {summary_path}")
print(f"Wrote {agg_path}")
PY

echo "Done: evaluate matrix → ${OUT_DIR}"

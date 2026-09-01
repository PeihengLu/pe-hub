#!/usr/bin/env bash
# Stage 03 — Evaluate registered weights on the held-out test partition.
#
# Writes append-only JSONL under results/ and a flat summary CSV.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh
#   RUN_ID=20260901T140000 ./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh
#
# ARC:
#   ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    export MODELS="${MODEL}"
    export BENCHMARKS="${BENCHMARK}"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${RESULTS_DIR}/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_JSONL="${OUT_DIR}/results.jsonl"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

print_benchmark_banner "03 Evaluate matrix (test holdout)"
echo "RUN_ID:  ${RUN_ID}"
echo "OUT_DIR: ${OUT_DIR}"
echo ""

PEEN_CMD=(python -m pe_ensemble.cli)
IDX=0
TOTAL=$(($(selected_models | wc -w) * $(selected_matrix_rows | wc -l)))

while IFS= read -r row; do
    IFS='|' read -r bench study dataset <<< "${row}"
    while read -r model; do
        [[ -n "${model}" ]] || continue
        IDX=$((IDX + 1))
        key="$(cell_key "${model}" "${bench}")"
        weights_state="weights__${key}"
        weights_id="$(read_state "${weights_state}")"

        SAFE_NAME="${model}__${bench}"
        STDOUT_FILE="${LOG_DIR}/${SAFE_NAME}.stdout"
        STDERR_FILE="${LOG_DIR}/${SAFE_NAME}.stderr"

        echo "[${IDX}/${TOTAL}] ${model} / ${weights_id} @ ${bench}"

        EVAL_ARGS=(
            evaluate
            --model "${model}"
            --weights "${weights_id}"
            --custom-benchmark
            --benchmark-name "${bench}"
            --study "${study}"
            --dataset "${dataset}"
            --device "${DEVICE}"
            --sync
        )
        split_args_for_cell
        EVAL_ARGS+=("${SPLIT_ARGS[@]}")

        META_JSON="$(python -c "
import json
print(json.dumps({
  'run_id': '''${RUN_ID}''',
  'model': '''${model}''',
  'weights': '''${weights_id}''',
  'benchmark_name': '''${bench}''',
  'study': '''${study}''',
  'dataset': '''${dataset}''',
  'split_strategy': 'holdout_3',
  'split_random_state': int('''${SPLIT_RANDOM_STATE}'''),
}))
")"

        set +e
        "${PEEN_CMD[@]}" "${EVAL_ARGS[@]}" > "${STDOUT_FILE}" 2> "${STDERR_FILE}"
        EXIT_CODE=$?
        set -e

        python - "${RESULTS_JSONL}" "${STDOUT_FILE}" "${STDERR_FILE}" "${EXIT_CODE}" "${META_JSON}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
stdout_path = Path(sys.argv[2])
stderr_path = Path(sys.argv[3])
exit_code = int(sys.argv[4])
meta = json.loads(sys.argv[5])

stdout = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()

record = dict(meta)
record["exit_code"] = exit_code

payload = None
if stdout:
    start = stdout.find("{")
    if start >= 0:
        try:
            payload = json.loads(stdout[start:])
        except json.JSONDecodeError:
            payload = None

if payload is not None:
    record.update(payload)
    metrics = payload.get("metrics") or {}
    if isinstance(metrics, dict):
        record["test_spearman"] = metrics.get("spearman")
        record["test_pearson"] = metrics.get("pearson")
        record["test_mse"] = metrics.get("mse")
    if payload.get("skipped"):
        record.setdefault("status", "skipped")
    elif payload.get("error_type") == "data_leak" or payload.get("status") == "error":
        record.setdefault("status", "error")
    elif payload.get("metrics") is not None:
        record.setdefault("status", "ok")
    else:
        record.setdefault("status", payload.get("status") or "unknown")
else:
    record["status"] = "error"
    record["error_type"] = "cli_failure"
    record["metrics"] = None

if stderr and record.get("status") == "error":
    record["stderr_tail"] = stderr[-2000:]

with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

print(json.dumps(record, indent=2))
PY
        echo ""
    done < <(selected_models | tr ' ' '\n')
done < <(selected_matrix_rows)

python - "${RESULTS_JSONL}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
rows = []
if jsonl.is_file():
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

fields = [
    "run_id", "model", "weights", "benchmark_name", "study", "dataset",
    "status", "test_spearman", "test_pearson", "test_mse", "exit_code",
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {csv_path} ({len(rows)} rows)")
PY

printf '%s\n' "${RUN_ID}" > "${RESULTS_DIR}/LATEST_RUN_ID"
echo "Done: evaluate matrix → ${OUT_DIR}"

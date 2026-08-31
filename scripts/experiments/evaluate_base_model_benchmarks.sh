#!/usr/bin/env bash
# Evaluate base vendor weights on pooled PE-DB benchmarks via peen evaluate --sync.
#
# Weights:
#   - deepprime / DeepPrime_base
#   - oped / pegRNA_Model_Merged_saved.order3_decoder_weights
#   - optiprime / base
#   - pridict2 / all CV fold runs (4 experiments × run_0..4)
#
# Benchmarks (pooled sheets under each study/dataset list):
#   minsepie-insert-pooled, deeppe-pooled, deepprime-clinvar,
#   pridict1-library1, pridict2-library-diverse, optiprime-lib-mmr, optiprime-lib-cv
#
# Usage:
#   conda activate pedb
#   DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
#
# Env:
#   DEVICE     compute device (default: auto)
#   RUN_ID     output run id (default: UTC timestamp)
#   OUT_ROOT   results root (default: <repo>/results/base_model_eval)
#   SMOKE=1    evaluate only first weight × first benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../hyperparameter/_common.sh
source "${REPO_ROOT}/scripts/hyperparameter/_common.sh"
require_peen

# Prefer module invocation: the ``peen`` console script can segfault under some
# shell/completion environments while ``python -m pe_ensemble.cli`` is reliable.
PEEN_CMD=(python -m pe_ensemble.cli)

DEVICE="${DEVICE:-auto}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/base_model_eval}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
RESULTS_JSONL="${OUT_DIR}/results.jsonl"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

print_experiment_banner "Base model evaluation (pooled benchmarks)"
echo "RUN_ID:    ${RUN_ID}"
echo "OUT_DIR:   ${OUT_DIR}"
echo "DEVICE:    ${DEVICE}"
echo "PEEN_CMD:  ${PEEN_CMD[*]}"
echo ""

# model|weights|experiment_id|cv_run
WEIGHTS=(
  "deepprime|DeepPrime_base|DeepPrime_base|"
  "oped|pegRNA_Model_Merged_saved.order3_decoder_weights|oped_merged|"
  "optiprime|base|optiprime_base|"
)

# August 2023 CV experiments only. December 2023 bundles are incomplete
# (config expects K562MLH1dn heads; statedict only has HEK/K562) — remigration
# is not possible (vendor sources already moved; no K562MLH1dn files on disk).
# Those runs remain registered but are excluded here; see
# tests/test_pridict2_weight_bundles.py (KNOWN_BROKEN_IDS).
PRIDICT2_EXPERIMENTS=(
  "pridict1_1__exp_2023-08-25_20-55-53"
  "pridict1_2__exp_2023-08-28_22-22-26"
)
# Multi-head vendor runs require a cell-type suffix; use HEK as the primary head
# for cross-benchmark base-model evaluation (CV mean/std over run_0..4).
for exp in "${PRIDICT2_EXPERIMENTS[@]}"; do
  for run in 0 1 2 3 4; do
    WEIGHTS+=("pridict2|${exp}__run_${run}__HEK|${exp}|${run}")
  done
done

# name|study|dataset1,dataset2,...
BENCHMARKS=(
  "minsepie-insert-pooled|minsepie|library-insert-set12,library-insert-18nt,library-insert-codon-variant,library-insert-codon-hek3"
  "deeppe-pooled|deeppe|deeppe-ht,deeppe-type,deeppe-position,deeppe-endo"
  "deepprime-clinvar|deepprime|deepprime-clinvar"
  "pridict1-library1|pridict1|library1"
  "pridict2-library-diverse|pridict2|library-diverse"
  "optiprime-lib-mmr|optiprime|lib-mmr"
  "optiprime-lib-cv|optiprime|lib-cv"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  WEIGHTS=("${WEIGHTS[0]}")
  BENCHMARKS=("${BENCHMARKS[0]}")
  echo "SMOKE=1: running ${#WEIGHTS[@]} weight(s) × ${#BENCHMARKS[@]} benchmark(s)"
  echo ""
fi

TOTAL=$((${#WEIGHTS[@]} * ${#BENCHMARKS[@]}))
echo "Matrix: ${#WEIGHTS[@]} weights × ${#BENCHMARKS[@]} benchmarks = ${TOTAL} evaluations"
echo ""

cat > "${OUT_DIR}/matrix.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "device": "${DEVICE}",
  "n_weights": ${#WEIGHTS[@]},
  "n_benchmarks": ${#BENCHMARKS[@]},
  "n_evaluations": ${TOTAL}
}
EOF

IDX=0
for weight_spec in "${WEIGHTS[@]}"; do
  IFS='|' read -r MODEL WEIGHTS_ID EXPERIMENT_ID CV_RUN <<< "${weight_spec}"
  for bench_spec in "${BENCHMARKS[@]}"; do
    IFS='|' read -r BENCH_NAME STUDY DATASETS_CSV <<< "${bench_spec}"
    IDX=$((IDX + 1))
    SAFE_NAME="$(echo "${MODEL}__${WEIGHTS_ID}__${BENCH_NAME}" | tr '/:' '__')"
    STDOUT_FILE="${LOG_DIR}/${SAFE_NAME}.stdout"
    STDERR_FILE="${LOG_DIR}/${SAFE_NAME}.stderr"

    echo "[${IDX}/${TOTAL}] ${MODEL} / ${WEIGHTS_ID} @ ${BENCH_NAME}"

    DATASET_ARGS=()
    IFS=',' read -ra DS_ARR <<< "${DATASETS_CSV}"
    for ds in "${DS_ARR[@]}"; do
      DATASET_ARGS+=(--dataset "${ds}")
    done

    # Build enrichment fields as a JSON object for merging with peen output.
    META_JSON="$(python -c "
import json
print(json.dumps({
  'model': '''${MODEL}''',
  'weights': '''${WEIGHTS_ID}''',
  'experiment_id': '''${EXPERIMENT_ID}''' or None,
  'cv_run': (int('''${CV_RUN}''') if '''${CV_RUN}'''.strip() != '' else None),
  'study': '''${STUDY}''',
  'datasets': '''${DATASETS_CSV}'''.split(','),
  'benchmark_name': '''${BENCH_NAME}''',
}))
")"

    set +e
    "${PEEN_CMD[@]}" evaluate \
      --model "${MODEL}" \
      --weights "${WEIGHTS_ID}" \
      --custom-benchmark \
      --benchmark-name "${BENCH_NAME}" \
      --study "${STUDY}" \
      "${DATASET_ARGS[@]}" \
      --sync \
      --device "${DEVICE}" \
      > "${STDOUT_FILE}" 2> "${STDERR_FILE}"
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
    # peen may print plugin lines to stderr; stdout should be JSON only.
    # Tolerate leading non-JSON lines by scanning for the first '{' .
    start = stdout.find("{")
    if start >= 0:
        try:
            payload = json.loads(stdout[start:])
        except json.JSONDecodeError:
            payload = None

if payload is not None:
    record.update(payload)
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
    record["n_samples"] = None
    record["stderr_tail"] = stderr[-2000:] if stderr else None
    if not record.get("model"):
        record["model"] = None
    if not record.get("weights"):
        record["weights"] = None

with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, default=str) + "\n")
PY

  done
done

echo ""
echo "Wrote ${RESULTS_JSONL}"
echo "Summarizing..."
python "${SCRIPT_DIR}/summarize_eval_results.py" "${RESULTS_JSONL}"
echo "Done. Results under ${OUT_DIR}"

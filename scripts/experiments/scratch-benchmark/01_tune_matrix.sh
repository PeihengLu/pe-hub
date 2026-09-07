#!/usr/bin/env bash
# Stage 01 — Nested Optuna + final train + eval for one seed (or all seeds locally).
#
# Uses scripts/experiments/datasheet-benchmark: holdout_3 × N_SEEDS (default 3)
# with N_TRIALS (default 10) Optuna trials per seed. Tuning already registers
# best weights (the "train" step). Eval runs in the same job.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#   MODELS=oped BENCHMARKS=deepprime-clinvar ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#   INDEX=0 MODELS=oped BENCHMARKS=pridict1-library1 ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#   SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#
# ARC (default: 63 short L40S jobs — one per model × dataset × seed):
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   MODEL=oped BENCHMARK=deepprime-clinvar ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   INDEX=0 MODEL=oped BENCHMARK=pridict1-library1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    export MODELS="${MODEL}"
    export BENCHMARKS="${BENCHMARK}"
fi

ensure_run_id
print_benchmark_banner "01 Tune+train+eval (datasheet-benchmark, ${N_SEEDS} seeds × ${N_TRIALS} trials)"
echo "RUN_ID: ${RUN_ID}"
if [[ -n "${INDEX:-}" ]]; then
    echo "INDEX:  ${INDEX} (single seed)"
fi
echo ""

while IFS= read -r row; do
    parse_matrix_row "${row}"
    while read -r model; do
        [[ -n "${model}" ]] || continue
        key="$(cell_key "${model}" "${MATRIX_BENCH}")"
        state_key="$(cell_done_key "done" "${key}")"

        if [[ "${SKIP_IF_DONE:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
            echo "SKIP_IF_DONE=1: already finished ${state_key}"
            continue
        fi

        export FIXED_HP_JSON="$(fixed_tune_hp_json "${model}")"

        echo "======================================"
        echo "TUNE+TRAIN+EVAL ${model} @ ${MATRIX_BENCH}${INDEX:+ (index ${INDEX})}"
        echo "======================================"

        if ! run_datasheet_benchmark_cell "${model}" 0; then
            echo "Error: datasheet-benchmark failed for ${key}" >&2
            exit 1
        fi

        write_state "${state_key}" "${RUN_ID}"
        echo ""
    done < <(selected_models | tr ' ' '\n')
done < <(selected_matrix_rows)

echo "Done: tune+train+eval (RUN_ID=${RUN_ID})."

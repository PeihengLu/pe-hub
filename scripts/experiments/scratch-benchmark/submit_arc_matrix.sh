#!/usr/bin/env bash
# Submit scratch-benchmark as separate 1-GPU SLURM jobs (one per seed).
#
# peen trains on a single GPU per job (Lightning devices=1). Default fan-out:
#   7 datasets × 3 models × N_SEEDS (3) = 63 jobs
# Each job: 10 Optuna trials + 1 final train + eval for that seed.
#
# Usage (from repo root, on htc-login):
#   ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh
#
# Pack all seeds of a cell into one job (legacy / medium walltime):
#   SUBMIT_SEEDS=0 ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh
#
# Env forwarded to each submit: ARC_PARTITION, ARC_TIME, ARC_GPU_CONSTRAINT, SMOKE, …
# Filter:
#   MODELS=oped BENCHMARKS=deepprime-clinvar ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARC_SUBMIT="${REPO_ROOT}/scripts/cluster/oxford-arc/submit.sh"
STAGE="${1:?Usage: $0 <01_tune_matrix.sh|02_train_matrix.sh>}"

# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ -n "${MODEL:-}" && -z "${MODELS:-}" ]]; then
    export MODELS="${MODEL}"
fi
if [[ -n "${BENCHMARK:-}" && -z "${BENCHMARKS:-}" ]]; then
    export BENCHMARKS="${BENCHMARK}"
fi

if [[ ! -x "${ARC_SUBMIT}" ]]; then
    echo "Error: ${ARC_SUBMIT} not found or not executable" >&2
    exit 1
fi

ensure_run_id
export RUN_ID N_SEEDS N_TRIALS PROTOCOL

MODELS_ARR=($(selected_models))
SUBMITTED=0
SUBMIT_SEEDS="${SUBMIT_SEEDS:-1}"
if [[ "${SUBMIT_SEEDS}" == "1" ]]; then
    JOB_UNIT="seed"
else
    JOB_UNIT="cell"
fi

echo "Submitting ${STAGE} — one 1-GPU job per ${JOB_UNIT}"
echo "models: $(selected_models)"
echo "RUN_ID: ${RUN_ID}"
echo "protocol: ${PROTOCOL} × ${N_SEEDS} seeds × ${N_TRIALS} trials"
echo ""

while IFS= read -r row; do
    IFS='|' read -r bench _study _dataset <<< "${row}"
    for model in "${MODELS_ARR[@]}"; do
        if [[ "${SUBMIT_SEEDS}" == "1" ]]; then
            for index in $(seq 0 $((N_SEEDS - 1))); do
                echo "--- ${model} @ ${bench} seed_index=${index} ---"
                _ARC_MATRIX_CELL=1 MODEL="${model}" BENCHMARK="${bench}" INDEX="${index}" RUN_ID="${RUN_ID}" \
                    "${ARC_SUBMIT}" "${STAGE}"
                SUBMITTED=$((SUBMITTED + 1))
                echo ""
            done
        else
            echo "--- ${model} @ ${bench} (all ${N_SEEDS} seeds) ---"
            _ARC_MATRIX_CELL=1 MODEL="${model}" BENCHMARK="${bench}" RUN_ID="${RUN_ID}" \
                "${ARC_SUBMIT}" "${STAGE}"
            SUBMITTED=$((SUBMITTED + 1))
            echo ""
        fi
    done
done < <(selected_matrix_rows)

echo "Submitted ${SUBMITTED} job(s) for ${STAGE} (RUN_ID=${RUN_ID})."
echo "After they finish: RUN_ID=${RUN_ID} ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh"

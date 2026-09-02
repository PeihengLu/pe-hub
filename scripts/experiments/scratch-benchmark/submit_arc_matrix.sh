#!/usr/bin/env bash
# Submit scratch-benchmark matrix cells as separate 1-GPU SLURM jobs (parallel).
#
# peen trains on a single GPU per job (Lightning devices=1). Request multiple
# L40S by submitting one job per model×benchmark cell, not multiple GPUs on one job.
#
# Usage (from repo root, on htc-login):
#   ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh
#   ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 02_train_matrix.sh
#   ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 03_evaluate_matrix.sh
#
# Env forwarded to each submit: ARC_PARTITION, ARC_TIME, ARC_GPU_CONSTRAINT, SMOKE, …
# Filter:
#   MODELS=oped BENCHMARKS=deepprime-clinvar ./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARC_SUBMIT="${REPO_ROOT}/scripts/cluster/oxford-arc/submit.sh"
STAGE="${1:?Usage: $0 <01_tune_matrix.sh|02_train_matrix.sh|03_evaluate_matrix.sh>}"

# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ ! -x "${ARC_SUBMIT}" ]]; then
    echo "Error: ${ARC_SUBMIT} not found or not executable" >&2
    exit 1
fi

MODELS_ARR=($(selected_models))
SUBMITTED=0

echo "Submitting ${STAGE} — one 1-GPU job per cell"
echo "models: $(selected_models)"
echo ""

while IFS= read -r row; do
    IFS='|' read -r bench _study _dataset <<< "${row}"
    for model in "${MODELS_ARR[@]}"; do
        echo "--- ${model} @ ${bench} ---"
        MODEL="${model}" BENCHMARK="${bench}" "${ARC_SUBMIT}" "${STAGE}"
        SUBMITTED=$((SUBMITTED + 1))
        echo ""
    done
done < <(selected_matrix_rows)

echo "Submitted ${SUBMITTED} job(s) for ${STAGE}."

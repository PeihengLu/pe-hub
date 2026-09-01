#!/usr/bin/env bash
# Run the full scratch-benchmark pipeline: tune → train → evaluate.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/run_all.sh
#   SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh
#   SKIP_IF_DONE=1 ./scripts/experiments/scratch-benchmark/run_all.sh
#
# ARC (sequential stages — submit each separately for long runs):
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   ./scripts/cluster/oxford-arc/submit.sh 02_train_matrix.sh
#   ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGES=(
    "${SCRIPT_DIR}/01_tune_matrix.sh"
    "${SCRIPT_DIR}/02_train_matrix.sh"
    "${SCRIPT_DIR}/03_evaluate_matrix.sh"
)

for stage in "${STAGES[@]}"; do
    echo ""
    echo "========== $(basename "${stage}") =========="
    bash "${stage}"
done

echo ""
echo "Pipeline complete."

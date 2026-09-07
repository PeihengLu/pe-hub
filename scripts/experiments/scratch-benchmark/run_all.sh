#!/usr/bin/env bash
# Run the scratch-benchmark pipeline: nested 10-trial × 3-seed tune+train+eval,
# then aggregate summaries.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/run_all.sh
#   SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh
#   SKIP_IF_DONE=1 ./scripts/experiments/scratch-benchmark/run_all.sh
#
# ARC (one short L40S job per seed; 03 is aggregation after they finish):
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   RUN_ID=<id> ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGES=(
    "${SCRIPT_DIR}/01_tune_matrix.sh"
    "${SCRIPT_DIR}/03_evaluate_matrix.sh"
)

for stage in "${STAGES[@]}"; do
    echo ""
    echo "========== $(basename "${stage}") =========="
    bash "${stage}"
done

echo ""
echo "Pipeline complete."

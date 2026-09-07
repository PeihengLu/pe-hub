#!/usr/bin/env bash
# Resume wrapper for 01 — same tune+train+eval job, skip finished seeds.
#
# There is no separate train stage: datasheet-benchmark already calls
# register_best_weights after Optuna. Re-queue 01 (or this wrapper) with the
# same RUN_ID / INDEX if a short job hits the 12h wall.
#
# Usage:
#   SKIP_IF_DONE=1 ./scripts/experiments/scratch-benchmark/02_train_matrix.sh
#   RUN_ID=<id> INDEX=0 MODEL=oped BENCHMARK=deepprime-clinvar \
#     ./scripts/cluster/oxford-arc/submit.sh 02_train_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
exec "${SCRIPT_DIR}/01_tune_matrix.sh"

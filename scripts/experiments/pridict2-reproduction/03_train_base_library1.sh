#!/usr/bin/env bash
# Stage 03 — Train PRIDICT2 base on library1 (uses HPO preset via merge mode).
# Writes state key: base_library1
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/03_train_base_library1.sh
#   SKIP_IF_DONE=1 SMOKE=1 DEVICE=mps ./.../03_train_base_library1.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

STATE_KEY="base_library1"
print_repro_banner "03 Train base: PRIDICT library1"
maybe_skip_if_state "${STATE_KEY}"

HP_JSON="${HYPERPARAMETERS_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${HP_JSON}" ]]; then
    HP_JSON='{"num_epochs":3,"batch_size":64}'
fi

TRAIN_ARGS=(
    train
    --model "${MODEL}"
    --dataset-name "${NAME_BASE_L1}"
    --study pridict1 --dataset library1
    --cell-line "${BASE_CELL_LINE}" --pe-system "${PE_SYSTEM}"
    --split-strategy holdout_3
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --device "${DEVICE}"
    --notes "pridict2-reproduction: base train on library1"
)

if [[ -n "${HP_JSON}" ]]; then
    TRAIN_ARGS+=(--hyperparameters-json "${HP_JSON}")
fi

run_peen_capture_weights "${STATE_KEY}" "${TRAIN_ARGS[@]}"

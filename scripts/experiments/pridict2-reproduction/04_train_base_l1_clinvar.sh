#!/usr/bin/env bash
# Stage 04 — Train PRIDICT2 base on library1 + DeepPrime ClinVar.
# Writes state key: base_l1_clinvar
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/04_train_base_l1_clinvar.sh
#   SKIP_IF_DONE=1 SMOKE=1 DEVICE=mps ./.../04_train_base_l1_clinvar.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

STATE_KEY="base_l1_clinvar"
print_repro_banner "04 Train base: library1 + DeepPrime ClinVar"
maybe_skip_if_state "${STATE_KEY}"

HP_JSON="${HYPERPARAMETERS_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${HP_JSON}" ]]; then
    HP_JSON='{"num_epochs":3,"batch_size":64}'
elif [[ -z "${HP_JSON}" ]]; then
    HP_JSON='{}'
fi
# ClinVar lacks the outcome trio → MSEloss (intended-edit only).
HP_JSON="$(force_mse_loss_json "${HP_JSON}")"

TRAIN_ARGS=(
    train
    --model "${MODEL}"
    --dataset-name "${NAME_BASE_L1C}"
    --study pridict1 --dataset library1
    --study deepprime --dataset deepprime-clinvar
    --cell-line "${BASE_CELL_LINE}" --pe-system "${PE_SYSTEM}"
    --merge
    --use-original-fold
    --original-fold-test-value=-1
    --split-strategy cv
    --cv-folds "${CV_FOLDS}"
    --test-pct "${TEST_PCT}"
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --device "${DEVICE}"
    --hyperparameters-json "${HP_JSON}"
    --notes "pridict2-reproduction: base train on L1+ClinVar; MSEloss"
)

run_peen_capture_weights "${STATE_KEY}" "${TRAIN_ARGS[@]}"

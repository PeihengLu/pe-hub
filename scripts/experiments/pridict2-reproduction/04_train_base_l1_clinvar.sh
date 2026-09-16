#!/usr/bin/env bash
# Stage 04 — Train PRIDICT2 base on library1 + DeepPrime ClinVar.
# holdout_3 + DeepPrime author fold -1 as test (same split family as stage 02).
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
print_repro_banner "04 Train base: library1 + DeepPrime ClinVar (holdout_3)"
maybe_skip_if_state "${STATE_KEY}"

HP_JSON="${HYPERPARAMETERS_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${HP_JSON}" ]]; then
    HP_JSON="$(smoke_fixed_hp_json)"
elif [[ -z "${HP_JSON}" ]]; then
    HP_JSON='{}'
fi
# ClinVar lacks the outcome trio — MSEloss (same as all PRIDICT2 stages).
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
    --split-strategy holdout_3
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --device "${DEVICE}"
    --hyperparameters-json "${HP_JSON}"
    --notes "pridict2-reproduction: base train on L1+ClinVar; holdout_3; MSEloss"
)

run_peen_capture_weights "${STATE_KEY}" "${TRAIN_ARGS[@]}"

#!/usr/bin/env bash
# Stage 06 — Fine-tune both bases on library-diverse HEK and K562 (4 models).
# Requires state: base_library1, base_l1_clinvar
# Writes: ft_base_library1_<cell>, ft_base_l1_clinvar_<cell>
#
# Protocol: author 5-fold CV (no outer random holdout), then evaluate each
# registered model on LD_TEST_FOLD (default 4).
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/06_finetune_transfer.sh
#   SKIP_IF_DONE=1 SMOKE=1 DEVICE=mps ./.../06_finetune_transfer.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "06 Fine-tune transfer on library-diverse (author CV5)"

BASE_L1="$(read_state base_library1)"
BASE_L1C="$(read_state base_l1_clinvar)"
echo "base_library1 weights:   ${BASE_L1}"
echo "base_l1_clinvar weights: ${BASE_L1C}"
echo "LD_TEST_FOLD (post-train evaluate): ${LD_TEST_FOLD}"
echo ""

HP_JSON="${HYPERPARAMETERS_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${HP_JSON}" ]]; then
    HP_JSON="$(smoke_fixed_hp_json)"
elif [[ -z "${HP_JSON}" ]]; then
    HP_JSON='{}'
fi
# library-diverse is edit-efficiency only.
HP_JSON="$(force_mse_loss_json "${HP_JSON}")"

# shellcheck disable=SC2206
lines=(${FT_CELL_LINES})

finetune_one() {
    local base_key="$1"
    local pretrained="$2"
    local cell="$3"
    local state_key="ft_${base_key}_${cell}"

    if [[ "${SKIP_IF_DONE:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
        echo "SKIP_IF_DONE=1: ${state_key}=$(cat "$(state_path "${state_key}")")"
        return 0
    fi

    local args=(
        train
        --model "${MODEL}"
        --dataset-name "${NAME_FT_PREFIX}-${base_key}-${cell}"
        --study pridict2 --dataset library-diverse
        --cell-line "${cell}" --pe-system "${PE_SYSTEM}"
        --device "${DEVICE}"
        --pretrained-weights "${pretrained}"
        --hyperparameters-json "${HP_JSON}"
        --notes "pridict2-reproduction: FT ${base_key} → library-diverse ${cell}; author CV5; MSEloss"
    )
    append_library_diverse_cv_args args

    run_peen_capture_weights "${state_key}" "${args[@]}"

    echo "---- evaluate ${state_key} on author fold ${LD_TEST_FOLD} ----"
    evaluate_library_diverse_test_fold \
        "$(cat "$(state_path "${state_key}")")" \
        "${cell}" \
        "${LD_TEST_FOLD}" \
        "${state_key}"
}

for cell in "${lines[@]}"; do
    echo "==== Fine-tune → ${cell} ===="
    finetune_one "base_library1" "${BASE_L1}" "${cell}"
    finetune_one "base_l1_clinvar" "${BASE_L1C}" "${cell}"
done

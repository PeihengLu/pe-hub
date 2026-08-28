#!/usr/bin/env bash
# Stage 02 — Optuna HPO for PRIDICT2 base on library1 + DeepPrime ClinVar.
# Uses --merge --use-original-fold so overlapping library1 loci inherit
# DeepPrime original_fold; remaining loci get random CV (+ outer test_pct).
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/02_tune_base_l1_clinvar.sh
#   SMOKE=1 DEVICE=mps ./.../02_tune_base_l1_clinvar.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

STUDY_NAME="${STUDY_NAME:-pridict2__repro_base_l1_clinvar}"
FIXED_HP_JSON="${FIXED_HP_JSON:-}"
# Merged study uses a composite preset key (study/dataset pairs joined with +).
DATASET_KEY="${DATASET_KEY:-pridict1/library1+deepprime/deepprime_clinvar/${BASE_CELL_LINE}/${PE_SYSTEM}}"
DATASET_KEY="$(echo "${DATASET_KEY}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

print_repro_banner "02 Tune base: library1 + DeepPrime ClinVar"
maybe_skip_if_tuned "${MODEL}" "${DATASET_KEY}"

FIXED_HP_JSON="${FIXED_HP_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON="$(smoke_fixed_hp_json)"
elif [[ -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON='{}'
fi
# ClinVar / merge path is edit-efficiency only — keep MSEloss explicit for Optuna.
FIXED_HP_JSON="$(force_mse_loss_json "${FIXED_HP_JSON}")"

TUNE_ARGS=(
    tune
    --model "${MODEL}"
    --dataset-name "${NAME_BASE_L1C}"
    --study pridict1 --dataset library1
    --study deepprime --dataset deepprime-clinvar
    --cell-line "${BASE_CELL_LINE}"
    --pe-system "${PE_SYSTEM}"
    --merge
    --use-original-fold
    --original-fold-test-value=-1
    --split-strategy cv
    --cv-folds "${CV_FOLDS}"
    --test-pct "${TEST_PCT}"
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --n-trials "${N_TRIALS}"
    --study-name "${STUDY_NAME}"
    --device "${DEVICE}"
    --register-best-weights
    --fixed-hyperparameters-json "${FIXED_HP_JSON}"
    --notes "pridict2-reproduction: L1+ClinVar base; DeepPrime folds by target_uid; MSEloss"
)

echo "+ peen ${TUNE_ARGS[*]}"
echo ""
exec peen "${TUNE_ARGS[@]}"

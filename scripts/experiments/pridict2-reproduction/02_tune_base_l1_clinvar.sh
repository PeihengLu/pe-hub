#!/usr/bin/env bash
# Stage 02 — Optuna HPO for PRIDICT2 base on library1 + DeepPrime ClinVar.
# holdout_3 (70/15/15), same protocol as stage 01/03 — not 5-fold CV.
# --merge --use-original-fold: ClinVar author fold -1 stays in test; overlapping
# library1 loci inherit DeepPrime original_fold via target_uid.
#
# Heavier than 01 (~338k rows) but one train/val per trial (not ×5 CV).
# On ARC: medium partition recommended; Optuna study resumes if re-queued.
#   ARC_PARTITION=medium ARC_TIME=48:00:00 \
#     ./scripts/cluster/oxford-arc/submit.sh 02_tune_base_l1_clinvar.sh
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/02_tune_base_l1_clinvar.sh
#   SMOKE=1 DEVICE=mps ./.../02_tune_base_l1_clinvar.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

export STUDY_NAME="${STUDY_NAME:-pridict2__repro_base_l1_clinvar_holdout3}"
# Merged study uses a composite preset key (study/dataset pairs joined with +).
DATASET_KEY="${DATASET_KEY:-pridict1/library1+deepprime/deepprime_clinvar/${BASE_CELL_LINE}/${PE_SYSTEM}}"
DATASET_KEY="$(echo "${DATASET_KEY}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

print_repro_banner "02 Tune base: library1 + DeepPrime ClinVar (holdout_3)"
maybe_skip_if_tuned "${MODEL}" "${DATASET_KEY}"

FIXED_HP_JSON="${FIXED_HP_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON="$(smoke_fixed_hp_json)"
elif [[ -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON='{}'
fi
# ClinVar / merge path is edit-efficiency only — keep MSEloss explicit for Optuna.
FIXED_HP_JSON="$(force_mse_loss_json "${FIXED_HP_JSON}")"
FIXED_HP_JSON="$(with_load_pretrained_json "${FIXED_HP_JSON}" false)"
export FIXED_HP_JSON

exec "${HP_DIR}/tune_hpo_holdout3.sh" \
    --model "${MODEL}" \
    --dataset-name "${NAME_BASE_L1C}" \
    --study pridict1 --dataset library1 \
    --study deepprime --dataset deepprime-clinvar \
    --cell-line "${BASE_CELL_LINE}" --pe-system "${PE_SYSTEM}" \
    --merge \
    --use-original-fold \
    --original-fold-test-value=-1

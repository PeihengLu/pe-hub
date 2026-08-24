#!/usr/bin/env bash
# Stage 01 — Optuna HPO for PRIDICT2 base on PRIDICT library1
# (random 5-fold CV + outer test; no author folds).
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh
#   SMOKE=1 SKIP_IF_TUNED=1 DEVICE=mps ./.../01_tune_base_library1.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

export STUDY_NAME="${STUDY_NAME:-pridict2__repro_base_library1}"
DATASET_KEY="${DATASET_KEY:-pridict1/library1/${BASE_CELL_LINE}/${PE_SYSTEM}}"
DATASET_KEY="$(echo "${DATASET_KEY}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

print_repro_banner "01 Tune base: PRIDICT library1"
maybe_skip_if_tuned "${MODEL}" "${DATASET_KEY}"

exec "${HP_DIR}/tune_hpo_cv5.sh" \
    --model "${MODEL}" \
    --dataset-name "${NAME_BASE_L1}" \
    --study pridict1 --dataset library1 \
    --cell-line "${BASE_CELL_LINE}" --pe-system "${PE_SYSTEM}"

#!/usr/bin/env bash
# Stage 05 — Optuna HPO for fine-tune stage on PRIDICT2 library-diverse.
# Runs once per cell line in FT_CELL_LINES (default: hek k562).
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/05_tune_finetune_library_diverse.sh
#   CELL_LINE=hek SMOKE=1 ./.../05_tune_finetune_library_diverse.sh   # single line

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "05 Tune fine-tune: library-diverse"

if [[ -n "${CELL_LINE:-}" ]]; then
    lines=("${CELL_LINE}")
else
    # shellcheck disable=SC2206
    lines=(${FT_CELL_LINES})
fi

for cell in "${lines[@]}"; do
    echo "---- library-diverse / ${cell} ----"
    STUDY_NAME="pridict2__repro_ft_library_diverse_${cell}" \
        "${HP_DIR}/tune_hpo_cv5.sh" \
        --model "${MODEL}" \
        --dataset-name "${NAME_FT_PREFIX}-library-diverse-${cell}" \
        --study pridict2 --dataset library-diverse \
        --cell-line "${cell}" --pe-system "${PE_SYSTEM}"
done

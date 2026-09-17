#!/usr/bin/env bash
# Stage 06 — Fine-tune Model B (L1+ClinVar base) on library-diverse.
# One holdout_3 train per author fold (0–4) × cell (hek / k562).
# Vendor-style run_x: fold x held out as test; no CV + final export.
#
# Requires state: base_l1_clinvar (stage 04)
# Writes: ft_base_l1_clinvar_<cell>_fold<fold>
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/06_finetune_transfer.sh
#   CELL_LINE=hek FOLD=0 ./.../06_finetune_transfer.sh
#   SKIP_IF_DONE=1 SMOKE=1 DEVICE=mps ./.../06_finetune_transfer.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "06 Fine-tune Model B: L1+ClinVar → library-diverse (fold holdouts)"

BASE_L1C="$(read_state base_l1_clinvar)"
echo "base_l1_clinvar weights: ${BASE_L1C}"
echo "cells: $(ft_cell_lines)"
echo "folds: $(ft_folds)"
echo ""

# shellcheck disable=SC2206
lines=($(ft_cell_lines))
# shellcheck disable=SC2206
folds=($(ft_folds))

for cell in "${lines[@]}"; do
    for fold in "${folds[@]}"; do
        echo "==== Model B FT → ${cell} fold ${fold} ===="
        finetune_fold_model "base_l1_clinvar" "${BASE_L1C}" "${cell}" "${fold}"
        echo ""
    done
done

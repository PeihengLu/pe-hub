#!/usr/bin/env bash
# Stage 05 — Fine-tune Model A (library1 base) on library-diverse.
# One holdout_3 train per author fold (0–4) × cell (hek / k562).
# Vendor-style run_x: fold x held out as test; no CV + final export.
#
# Requires state: base_library1 (stage 03)
# Writes: ft_base_library1_<cell>_fold<fold>
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/05_tune_finetune_library_diverse.sh
#   CELL_LINE=hek FOLD=0 ./.../05_tune_finetune_library_diverse.sh
#   SKIP_IF_DONE=1 SMOKE=1 DEVICE=mps ./.../05_tune_finetune_library_diverse.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "05 Fine-tune Model A: library1 → library-diverse (fold holdouts)"

BASE_L1="$(read_state base_library1)"
echo "base_library1 weights: ${BASE_L1}"
echo "cells: $(ft_cell_lines)"
echo "folds: $(ft_folds)"
echo ""

# shellcheck disable=SC2206
lines=($(ft_cell_lines))
# shellcheck disable=SC2206
folds=($(ft_folds))

for cell in "${lines[@]}"; do
    for fold in "${folds[@]}"; do
        echo "==== Model A FT → ${cell} fold ${fold} ===="
        finetune_fold_model "base_library1" "${BASE_L1}" "${cell}" "${fold}"
        echo ""
    done
done

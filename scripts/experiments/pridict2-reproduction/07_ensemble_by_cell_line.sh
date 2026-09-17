#!/usr/bin/env bash
# Stage 07 — Mean-ensemble Model A + Model B per cell × author fold.
# Requires: ft_base_library1_<cell>_fold<f>, ft_base_l1_clinvar_<cell>_fold<f>
# Evaluates each ensemble on the matching held-out author fold.
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/07_ensemble_by_cell_line.sh
#   CELL_LINE=hek FOLD=4 DEVICE=mps ./.../07_ensemble_by_cell_line.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "07 Ensemble Model A + B per cell × fold (mean)"

# shellcheck disable=SC2206
lines=($(ft_cell_lines))
# shellcheck disable=SC2206
folds=($(ft_folds))

for cell in "${lines[@]}"; do
    for fold in "${folds[@]}"; do
        w_a="$(read_state "ft_base_library1_${cell}_fold${fold}")"
        w_b="$(read_state "ft_base_l1_clinvar_${cell}_fold${fold}")"
        ens_name="pridict2-repro-ensemble-${cell}-fold${fold}"
        state_key="ensemble_${cell}_fold${fold}"

        if [[ "${SKIP_IF_DONE:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
            echo "SKIP_IF_DONE=1: ${state_key} already done"
            continue
        fi

        echo "==== Ensemble ${cell} fold ${fold} ===="
        echo "  member Model A (library1 base FT):   ${w_a}"
        echo "  member Model B (L1+ClinVar base FT): ${w_b}"

        ARGS=(
            ensemble
            --ensemble-name "${ens_name}"
            --combine mean
            --member "${MODEL}:${w_a}"
            --member "${MODEL}:${w_b}"
            --study pridict2 --dataset library-diverse
            --cell-line "${cell}" --pe-system "${PE_SYSTEM}"
            --split-strategy holdout_2
            --use-original-fold
            --original-fold-test-value "${fold}"
            --split-random-state "${SPLIT_RANDOM_STATE}"
            --device "${DEVICE}"
            --sync
        )

        logfile="$(state_path "${state_key}.log")"
        echo "+ peen ${ARGS[*]}"
        echo "  (log: ${logfile})"
        peen "${ARGS[@]}" 2>&1 | tee "${logfile}"
        write_state "${state_key}" "${ens_name}"
        echo ""
    done
done

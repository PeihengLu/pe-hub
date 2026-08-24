#!/usr/bin/env bash
# Stage 07 — Mean-ensemble the two fine-tuned models per cell line.
# Requires: ft_base_library1_<cell>, ft_base_l1_clinvar_<cell>
# Evaluates on library-diverse for that cell line (random holdout; no author folds).
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/07_ensemble_by_cell_line.sh
#   DEVICE=mps ./.../07_ensemble_by_cell_line.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "07 Ensemble by cell line (mean)"

# shellcheck disable=SC2206
lines=(${FT_CELL_LINES})

for cell in "${lines[@]}"; do
    w90="$(read_state "ft_base_library1_${cell}")"
    w390="$(read_state "ft_base_l1_clinvar_${cell}")"
    ens_name="pridict2-repro-ensemble-${cell}"
    state_key="ensemble_${cell}"

    if [[ "${SKIP_IF_DONE:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
        echo "SKIP_IF_DONE=1: ${state_key} already done"
        continue
    fi

    echo "==== Ensemble ${cell} ===="
    echo "  member library1-base FT:   ${w90}"
    echo "  member L1+ClinVar-base FT: ${w390}"

    ARGS=(
        ensemble
        --ensemble-name "${ens_name}"
        --combine mean
        --member "${MODEL}:${w90}"
        --member "${MODEL}:${w390}"
        --study pridict2 --dataset library-diverse
        --cell-line "${cell}" --pe-system "${PE_SYSTEM}"
        --split-strategy holdout_2
        --train-pct 0.8 --test-pct 0.2
        --no-use-original-fold
        --split-random-state "${SPLIT_RANDOM_STATE}"
        --device "${DEVICE}"
        --sync
    )

    logfile="$(state_path "${state_key}.log")"
    echo "+ peen ${ARGS[*]}"
    peen "${ARGS[@]}" 2>&1 | tee "${logfile}"
    write_state "${state_key}" "${ens_name}"
    echo ""
done

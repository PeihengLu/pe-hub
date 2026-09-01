#!/usr/bin/env bash
# Stage 01 — Optuna HPO (holdout_3) for each model × benchmark cell.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#   MODELS=oped BENCHMARKS=deepprime-clinvar ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#   SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
#
# ARC:
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   MODEL=oped BENCHMARK=deepprime-clinvar ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

# Single-cell shortcut for ARC (MODEL + BENCHMARK env).
if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    export MODELS="${MODEL}"
    export BENCHMARKS="${BENCHMARK}"
fi

print_benchmark_banner "01 Tune matrix (holdout_3 HPO)"

while IFS= read -r row; do
    IFS='|' read -r bench study dataset <<< "${row}"
    while read -r model; do
        [[ -n "${model}" ]] || continue
        key="$(cell_key "${model}" "${bench}")"
        state_key="tuned__${key}"

        if [[ "${SKIP_IF_DONE:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
            echo "SKIP_IF_DONE=1: already tuned ${key}"
            continue
        fi

        preset_key="$(dataset_preset_key_for_cell "${study}" "${dataset}")"
        fixed_hp="$(fixed_tune_hp_json "${model}")"

        echo "======================================"
        echo "TUNE ${model} @ ${bench}"
        echo "======================================"

        TUNE_ARGS=(
            --model "${model}"
            --dataset-name "$(dataset_name_for_cell "${model}" "${bench}")"
            --study "${study}"
            --dataset "${dataset}"
        )

        split_args_for_cell
        TUNE_ARGS+=("${SPLIT_ARGS[@]}")

        export STUDY_NAME="$(study_name_for_cell "${model}" "${bench}")"
        export DATASET_KEY="${preset_key}"
        export FIXED_HP_JSON="${fixed_hp}"

        if ! "${HP_DIR}/tune_hpo_holdout3.sh" "${TUNE_ARGS[@]}"; then
            echo "Error: tuning failed for ${key}" >&2
            exit 1
        fi

        write_state "${state_key}" "ok"
        echo ""
    done < <(selected_models | tr ' ' '\n')
done < <(selected_matrix_rows)

echo "Done: tune matrix."

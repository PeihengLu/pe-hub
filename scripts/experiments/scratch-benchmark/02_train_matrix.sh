#!/usr/bin/env bash
# Stage 02 — Final from-scratch train (merge tuned preset) + register weights.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/02_train_matrix.sh
#   SKIP_IF_DONE=1 ./scripts/experiments/scratch-benchmark/02_train_matrix.sh
#
# ARC:
#   ./scripts/cluster/oxford-arc/submit.sh 02_train_matrix.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    export MODELS="${MODEL}"
    export BENCHMARKS="${BENCHMARK}"
fi

print_benchmark_banner "02 Train matrix (merge preset, register weights)"

while IFS= read -r row; do
    IFS='|' read -r bench study dataset <<< "${row}"
    while read -r model; do
        [[ -n "${model}" ]] || continue
        key="$(cell_key "${model}" "${bench}")"
        state_key="weights__${key}"

        maybe_skip_if_state "${state_key}"

        tune_marker="tuned__${key}"
        if [[ ! -f "$(state_path "${tune_marker}")" ]]; then
            echo "Warning: no tune marker for ${key}; training with shipped/default preset only." >&2
        fi

        hp_json="$(train_hp_json "${model}")"
        if [[ "${SMOKE:-0}" == "1" && -z "${HYPERPARAMETERS_JSON:-}" ]]; then
            hp_json="$(smoke_fixed_hp_json)"
        elif [[ -n "${HYPERPARAMETERS_JSON:-}" ]]; then
            hp_json="${HYPERPARAMETERS_JSON}"
        fi

        echo "======================================"
        echo "TRAIN ${model} @ ${bench}"
        echo "======================================"

        TRAIN_ARGS=(
            train
            --model "${model}"
            --dataset-name "$(dataset_name_for_cell "${model}" "${bench}")"
            --study "${study}"
            --dataset "${dataset}"
            --hyperparameter-mode merge
            --hyperparameters-json "${hp_json}"
            --register-best-weights
            --device "${DEVICE}"
            --notes "${EXP_PREFIX}: train ${model} on ${bench} (holdout_3, from scratch)"
        )

        split_args_for_cell
        TRAIN_ARGS+=("${SPLIT_ARGS[@]}")

        run_peen_capture_weights "${state_key}" "${TRAIN_ARGS[@]}"
        echo ""
    done < <(selected_models | tr ' ' '\n')
done < <(selected_matrix_rows)

echo "Done: train matrix."

#!/usr/bin/env bash
# Shared helpers for the scratch-benchmark experiment:
#   model × dataset matrix via scripts/experiments/datasheet-benchmark
#   holdout_3 × N_SEEDS (default 3) with Optuna N_TRIALS (default 10) per seed.
#
# Each cell: tune → register best weights (train) → evaluate test.
# Benchmarks align with scripts/experiments/evaluate_base_model_benchmarks.sh.
# Models: deepprime, oped, pridict2 (OptiPrime model excluded; lib-* are datasets).

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_DIR="$(cd "${EXP_DIR}/../../hyperparameter" && pwd)"
BENCH_PY="${EXP_DIR}/../datasheet-benchmark/run_benchmark.py"

# Nested protocol defaults — set before hyperparameter/_common.sh so its
# N_TRIALS="${N_TRIALS:-20}" keeps 10.
: "${N_SEEDS:=3}"
: "${N_TRIALS:=10}"
: "${PROTOCOL:=holdout_3}"

# shellcheck source=../../hyperparameter/_common.sh
source "${HP_DIR}/_common.sh"

if [[ "${SMOKE:-0}" == "1" ]]; then
    N_SEEDS="${N_SEEDS_SMOKE:-2}"
fi
N="${N_SEEDS}"

STATE_DIR="${STATE_DIR:-${EXP_DIR}/state}"
RESULTS_DIR="${RESULTS_DIR:-${EXP_DIR}/results}"
mkdir -p "${STATE_DIR}" "${RESULTS_DIR}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    ARC_SMOKE_DIR="$(cd "${EXP_DIR}/../../cluster/oxford-arc" && pwd)"
    # shellcheck source=../../cluster/oxford-arc/_smoke_common.sh
    source "${ARC_SMOKE_DIR}/_smoke_common.sh"
    setup_smoke_mini_data_root
    STATE_DIR="${WORK_DIR}/state"
    RESULTS_DIR="${WORK_DIR}/results"
    mkdir -p "${STATE_DIR}" "${RESULTS_DIR}"
fi

EXP_PREFIX="${EXP_PREFIX:-scratch-benchmark}"
SPLIT_RANDOM_STATE="${SPLIT_RANDOM_STATE:-42}"
TRAIN_PCT="${TRAIN_PCT:-0.7}"
VAL_PCT="${VAL_PCT:-0.15}"
TEST_PCT="${TEST_PCT:-0.15}"
NUM_WORKERS="${NUM_WORKERS:-15}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-12}"
MAX_EPOCHS_DEEPPRIME="${MAX_EPOCHS_DEEPPRIME:-50}"
MAX_EPOCHS_OPED="${MAX_EPOCHS_OPED:-50}"
MAX_EPOCHS_PRIDICT2="${MAX_EPOCHS_PRIDICT2:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# bench_name|study|datasets_csv  (comma-separated when pooled under one study)
MATRIX_ALL=(
    "pridict1-library1|pridict1|library1"
    "pridict2-library-diverse|pridict2|library-diverse"
    "deepprime-clinvar|deepprime|deepprime-clinvar"
    "deeppe-pooled|deeppe|deeppe-ht,deeppe-type,deeppe-position,deeppe-endo"
    "minsepie-insert-pooled|minsepie|library-insert-set12,library-insert-18nt,library-insert-codon-variant,library-insert-codon-hek3"
    "optiprime-lib-mmr|optiprime|lib-mmr"
    "optiprime-lib-cv|optiprime|lib-cv"
)

MODELS_ALL=(deepprime oped pridict2)

state_path() {
    echo "${STATE_DIR}/$1"
}

write_state() {
    local key="$1"
    local value="$2"
    printf '%s\n' "${value}" > "$(state_path "${key}")"
    echo "Wrote state ${key}=${value}"
}

read_state() {
    local key="$1"
    local path
    path="$(state_path "${key}")"
    if [[ ! -f "${path}" ]]; then
        echo "Error: missing state '${key}' at ${path}" >&2
        echo "Run the earlier pipeline stage first (or set STATE_DIR)." >&2
        exit 1
    fi
    cat "${path}"
}

maybe_skip_if_state() {
    local key="$1"
    if [[ "${SKIP_IF_DONE:-0}" != "1" ]]; then
        return 0
    fi
    if [[ -f "$(state_path "${key}")" ]]; then
        echo "SKIP_IF_DONE=1: state already exists for ${key}=$(cat "$(state_path "${key}")")"
        exit 0
    fi
}

ensure_run_id() {
    if [[ -z "${RUN_ID:-}" ]]; then
        RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    printf '%s\n' "${RUN_ID}" > "${RESULTS_DIR}/LATEST_RUN_ID"
    export RUN_ID
}

cell_out_dir() {
    local model="$1"
    local bench="$2"
    echo "${RESULTS_DIR}/${RUN_ID}/$(cell_key "${model}" "${bench}")"
}

cell_done_key() {
    local prefix="$1"
    local key="$2"
    if [[ -n "${INDEX:-}" ]]; then
        echo "${prefix}__${key}__s${INDEX}"
    else
        echo "${prefix}__${key}"
    fi
}

run_datasheet_benchmark_cell() {
    local model="$1"
    local skip_eval="${2:-0}"
    local extra_args=("${@:3}")
    local py out_dir
    py="$(command -v python 2>/dev/null || command -v python3)"
    out_dir="$(cell_out_dir "${model}" "${MATRIX_BENCH}")"
    mkdir -p "${out_dir}"

    local args=(
        --model "${model}"
        --n "${N_SEEDS}"
        --n-trials "${N_TRIALS}"
        --protocol "${PROTOCOL}"
        --dataset-name "$(dataset_name_for_cell "${model}" "${MATRIX_BENCH}")"
        --device "${DEVICE}"
        --base-seed "${SPLIT_RANDOM_STATE}"
        --train-pct "${TRAIN_PCT}"
        --val-pct "${VAL_PCT}"
        --test-pct "${TEST_PCT}"
        --run-id "${RUN_ID}"
        --out-dir "${out_dir}"
        --merge
        --no-write-preset
    )
    append_study_dataset_args args "${MATRIX_STUDY}" "${MATRIX_DATASETS}"

    if [[ -n "${INDEX:-}" ]]; then
        args+=(--index "${INDEX}")
    fi
    if [[ "${SKIP_IF_DONE:-0}" == "1" ]]; then
        args+=(--skip-existing)
    fi
    if [[ "${skip_eval}" == "1" ]]; then
        args+=(--skip-eval)
    fi
    if [[ -n "${FIXED_HP_JSON:-}" ]]; then
        args+=(--fixed-hyperparameters-json "${FIXED_HP_JSON}")
    fi
    if ((${#extra_args[@]})); then
        args+=("${extra_args[@]}")
    fi

    echo "+ ${py} ${BENCH_PY} ${args[*]}"
    "${py}" "${BENCH_PY}" "${args[@]}"
}

cell_key() {
    local model="$1"
    local bench="$2"
    echo "${model}__${bench}"
}

parse_matrix_row() {
    local row="$1"
    IFS='|' read -r MATRIX_BENCH MATRIX_STUDY MATRIX_DATASETS <<< "${row}"
    : "${MATRIX_BENCH:?invalid matrix row: ${row}}"
    : "${MATRIX_STUDY:?invalid matrix row: ${row}}"
    : "${MATRIX_DATASETS:?invalid matrix row: ${row}}"
}

append_study_dataset_args() {
    local -n _out="$1"
    local study="$2"
    local datasets_csv="$3"
    local ds
    _out+=(--study "${study}")
    IFS=',' read -ra _ds_arr <<< "${datasets_csv}"
    for ds in "${_ds_arr[@]}"; do
        ds="${ds#"${ds%%[![:space:]]*}"}"
        ds="${ds%"${ds##*[![:space:]]}"}"
        [[ -n "${ds}" ]] || continue
        _out+=(--dataset "${ds}")
    done
}

datasets_display_for_row() {
    local datasets_csv="$1"
    echo "${datasets_csv//,/, }"
}

dataset_name_for_cell() {
    local model="$1"
    local bench="$2"
    echo "${EXP_PREFIX}__${model}__${bench}"
}

fixed_tune_hp_json() {
    local model="$1"
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    MODEL="${model}" "${py}" - <<'PY'
import json, os
model = os.environ["MODEL"].strip().lower()
hp = {"load_pretrained": False}
if model == "pridict2":
    hp["loss_func"] = "MSEloss"
    hp["y_ref"] = ["averageedited"]
print(json.dumps(hp, separators=(",", ":")))
PY
}

selected_models() {
    if [[ -n "${MODELS:-}" ]]; then
        # shellcheck disable=SC2206
        echo "${MODELS}"
        return
    fi
    echo "${MODELS_ALL[*]}"
}

selected_matrix_rows() {
    local want_bench bench_row
    if [[ -n "${BENCHMARKS:-}" ]]; then
        for want_bench in ${BENCHMARKS}; do
            for bench_row in "${MATRIX_ALL[@]}"; do
                IFS='|' read -r bench _study _dataset <<< "${bench_row}"
                if [[ "${bench}" == "${want_bench}" ]]; then
                    echo "${bench_row}"
                fi
            done
        done
        return
    fi
    printf '%s\n' "${MATRIX_ALL[@]}"
}

print_benchmark_banner() {
    local title="$1"
    print_experiment_banner "${title}"
    echo "STATE_DIR:   ${STATE_DIR}"
    echo "RESULTS_DIR: ${RESULTS_DIR}"
    echo "EXP_PREFIX:  ${EXP_PREFIX}"
    echo "protocol:    ${PROTOCOL} × ${N_SEEDS} seeds, ${N_TRIALS} Optuna trials/seed"
    echo "split:       holdout_3 (${TRAIN_PCT}/${VAL_PCT}/${TEST_PCT}, base_seed=${SPLIT_RANDOM_STATE})"
    echo "models:      $(selected_models)"
    echo "benchmarks:"
    local row
    while IFS= read -r row; do
        parse_matrix_row "${row}"
        echo "  ${MATRIX_BENCH} (${MATRIX_STUDY}: $(datasets_display_for_row "${MATRIX_DATASETS}") )"
    done < <(selected_matrix_rows)
    if [[ "${SMOKE:-0}" == "1" ]]; then
        echo "SMOKE:       1 (n_trials=${N_TRIALS} n_seeds=${N_SEEDS}; mini data unless SMOKE_FULL_DATA=1)"
    fi
    echo ""
}

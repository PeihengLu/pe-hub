#!/usr/bin/env bash
# Shared helpers for the scratch-benchmark experiment:
#   model × dataset matrix, holdout_3 tune → train → evaluate (from scratch).
#
# Benchmarks align with scripts/experiments/evaluate_base_model_benchmarks.sh:
#   pridict1-library1, pridict2-library-diverse, deepprime-clinvar,
#   deeppe-pooled, minsepie-insert-pooled, optiprime-lib-mmr, optiprime-lib-cv
# Models: deepprime, oped, pridict2 (OptiPrime model excluded; lib-* are datasets).

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_DIR="$(cd "${EXP_DIR}/../../hyperparameter" && pwd)"
# shellcheck source=../../hyperparameter/_common.sh
source "${HP_DIR}/_common.sh"

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

extract_weights_id() {
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    "${py}" - <<'PY'
import json, sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
last = None
idx = 0
while True:
    start = text.find("{", idx)
    if start < 0:
        break
    try:
        obj, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        idx = start + 1
        continue
    if isinstance(obj, dict) and obj.get("weights_id"):
        last = obj["weights_id"]
    idx = end
if not last:
    sys.stderr.write("Error: no weights_id found in peen output\n")
    sys.exit(1)
print(last)
PY
}

run_peen_capture_weights() {
    local state_key="$1"
    shift
    local logfile
    logfile="$(state_path "${state_key}.log")"
    echo "+ peen $*"
    echo "  (log: ${logfile})"
    if peen "$@" 2>&1 | tee "${logfile}" | extract_weights_id > "$(state_path "${state_key}.tmp")"; then
        mv "$(state_path "${state_key}.tmp")" "$(state_path "${state_key}")"
        echo "Wrote state ${state_key}=$(cat "$(state_path "${state_key}")")"
    else
        rm -f "$(state_path "${state_key}.tmp")"
        echo "Error: peen failed; see ${logfile}" >&2
        exit 1
    fi
}

cell_key() {
    local model="$1"
    local bench="$2"
    echo "${model}__${bench}"
}

dataset_preset_key_for_cell() {
    local study="$1"
    local datasets_csv="$2"
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    STUDY="${study}" DATASETS_CSV="${datasets_csv}" REPO_ROOT="${HP_DIR}/../.." \
        "${py}" - <<'PY'
import os, sys
from pathlib import Path

root = Path(os.environ["REPO_ROOT"]) / "services" / "pe-ensemble"
sys.path.insert(0, str(root))
from app.training.dataset_key import dataset_preset_key

study = os.environ["STUDY"]
datasets = [part.strip() for part in os.environ["DATASETS_CSV"].split(",") if part.strip()]
if not datasets:
    raise SystemExit("empty datasets list")
payload = datasets[0] if len(datasets) == 1 else datasets
key = dataset_preset_key(study=study, dataset=payload)
if not key:
    raise SystemExit(f"could not derive preset key for {study!r} / {datasets!r}")
print(key)
PY
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

study_name_for_cell() {
    local model="$1"
    local bench="$2"
    echo "${EXP_PREFIX}__${model}__${bench}"
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

train_hp_json() {
    local model="$1"
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    MODEL="${model}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
    MAX_EPOCHS_DEEPPRIME="${MAX_EPOCHS_DEEPPRIME}" \
    MAX_EPOCHS_OPED="${MAX_EPOCHS_OPED}" \
    MAX_EPOCHS_PRIDICT2="${MAX_EPOCHS_PRIDICT2}" \
    "${py}" - <<'PY'
import json, os
model = os.environ["MODEL"].strip().lower()
hp = {
    "load_pretrained": False,
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "num_workers": int(os.environ["NUM_WORKERS"]),
    "early_stopping_patience": int(os.environ["EARLY_STOPPING_PATIENCE"]),
    "freezing": False,
}
if model == "deepprime":
    n = int(os.environ["MAX_EPOCHS_DEEPPRIME"])
    hp["epochs"] = n
elif model == "oped":
    n = int(os.environ["MAX_EPOCHS_OPED"])
    hp["epoch_num"] = n
elif model == "pridict2":
    n = int(os.environ["MAX_EPOCHS_PRIDICT2"])
    hp["num_epochs"] = n
    hp["loss_func"] = "MSEloss"
    hp["y_ref"] = ["averageedited"]
else:
    raise SystemExit(f"unsupported model: {model}")
print(json.dumps(hp, separators=(",", ":")))
PY
}

split_args_for_cell() {
    SPLIT_ARGS=(
        --split-strategy holdout_3
        --train-pct "${TRAIN_PCT}"
        --val-pct "${VAL_PCT}"
        --test-pct "${TEST_PCT}"
        --split-random-state "${SPLIT_RANDOM_STATE}"
        --no-use-original-fold
    )
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
    local want_bench model bench_row
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
    echo "split:       holdout_3 (${TRAIN_PCT}/${VAL_PCT}/${TEST_PCT}, seed=${SPLIT_RANDOM_STATE})"
    echo "models:      $(selected_models)"
    echo "benchmarks:"
    local row bench
    while IFS= read -r row; do
        parse_matrix_row "${row}"
        IFS='|' read -r bench _study _datasets <<< "${row}"
        echo "  ${bench} (${_study}: $(datasets_display_for_row "${_datasets}"), holdout seed=${SPLIT_RANDOM_STATE})"
    done < <(selected_matrix_rows)
    if [[ "${SMOKE:-0}" == "1" ]]; then
        echo "SMOKE:       1 (n_trials=${N_TRIALS}; mini data unless SMOKE_FULL_DATA=1)"
    fi
    echo ""
}

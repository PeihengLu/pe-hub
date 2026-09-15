#!/usr/bin/env bash
# Shared helpers for PRIDICT 2.0 transfer + ensemble reproduction.
#
# Pipeline (Mathis et al. via PE Ensemble / PE-DB):
#   1. Base train on PRIDICT library1
#   2. Base train on library1 + DeepPrime ClinVar (DeepPrime folds on overlaps)
#   3. Fine-tune both bases on library-diverse HEK and K562 → 4 models
#   4. Mean-ensemble per cell line

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_DIR="$(cd "${EXP_DIR}/../../hyperparameter" && pwd)"
# shellcheck source=../../hyperparameter/_common.sh
source "${HP_DIR}/_common.sh"

STATE_DIR="${STATE_DIR:-${EXP_DIR}/state}"
mkdir -p "${STATE_DIR}"

# SMOKE=1: subsampled DATA_ROOT + isolated artifacts (local pre-ARC check).
# SMOKE_FULL_DATA=1: keep real DATA_ROOT (e.g. one trial on ARC full data).
if [[ "${SMOKE:-0}" == "1" ]]; then
    ARC_SMOKE_DIR="$(cd "${EXP_DIR}/../../cluster/oxford-arc" && pwd)"
    # shellcheck source=../../cluster/oxford-arc/_smoke_common.sh
    source "${ARC_SMOKE_DIR}/_smoke_common.sh"
    setup_smoke_mini_data_root
    STATE_DIR="${WORK_DIR}/state"
    mkdir -p "${STATE_DIR}"
fi

MODEL="${MODEL:-pridict2}"
PE_SYSTEM="${PE_SYSTEM:-pe2}"
BASE_CELL_LINE="${BASE_CELL_LINE:-hek293t}"
FT_CELL_LINES="${FT_CELL_LINES:-hek k562}"

# library-diverse author folds are 0..4. After pure CV training, peen’s final
# export holds out the last fold for early stopping — evaluate on that fold.
LD_TEST_FOLD="${LD_TEST_FOLD:-4}"

# Stable dataset-name labels (also used as state keys).
NAME_BASE_L1="${NAME_BASE_L1:-pridict2-repro-base-library1}"
NAME_BASE_L1C="${NAME_BASE_L1C:-pridict2-repro-base-l1-clinvar}"
NAME_FT_PREFIX="${NAME_FT_PREFIX:-pridict2-repro-ft}"

# PRIDICT2 uses a single edit-efficiency head (MSEloss on averageedited).
# pe_db maps editing_efficiency → averageedited at format time.
force_mse_loss_json() {
    local raw="${1-}"
    if [[ -z "${raw}" ]]; then
        raw="{}"
    fi
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    PE_HUB_HP_JSON_IN="${raw}" "${py}" - <<'PY'
import json
import os

raw = (os.environ.get("PE_HUB_HP_JSON_IN") or "{}").strip() or "{}"
data = json.loads(raw)
data["loss_func"] = "MSEloss"
data["y_ref"] = ["averageedited"]
print(json.dumps(data, separators=(",", ":")))
PY
}

# Merge load_pretrained into a hyperparameters JSON object (scratch vs transfer).
with_load_pretrained_json() {
    local raw="${1-}"
    local flag="${2:?usage: with_load_pretrained_json <json> true|false}"
    if [[ -z "${raw}" ]]; then
        raw="{}"
    fi
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    PE_HUB_HP_JSON_IN="${raw}" PE_HUB_LOAD_PRETRAINED="${flag}" "${py}" - <<'PY'
import json
import os

raw = (os.environ.get("PE_HUB_HP_JSON_IN") or "{}").strip() or "{}"
flag = (os.environ.get("PE_HUB_LOAD_PRETRAINED") or "").strip().lower()
data = json.loads(raw)
data["load_pretrained"] = flag in ("1", "true", "yes", "on")
if not data["load_pretrained"]:
    data.pop("weights", None)
print(json.dumps(data, separators=(",", ":")))
PY
}

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

# Extract the last JSON object's weights_id from a peen log file.
# Must take a file path: `python - <<EOF` would consume stdin as the program,
# so a piped/redirected log is never read (always "no weights_id").
extract_weights_id() {
    local log_file="${1:?usage: extract_weights_id <logfile>}"
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    if [[ -z "${py}" ]]; then
        echo "Error: python not found for extract_weights_id" >&2
        return 1
    fi
    if [[ ! -f "${log_file}" ]]; then
        echo "Error: peen log not found: ${log_file}" >&2
        return 1
    fi
    "${py}" -c '
import json, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
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
' "${log_file}"
}

# Newest peen train job dir (for debugging empty capture logs).
latest_train_job_dir() {
    local root="${TRAINING_JOBS_ROOT:-}"
    if [[ -z "${root}" ]]; then
        root="${EXP_DIR}/../../../services/pe-ensemble/jobs"
    fi
    root="$(cd "${root}" 2>/dev/null && pwd || true)"
    [[ -n "${root}" && -d "${root}" ]] || return 0
    local newest=""
    newest="$(ls -td "${root}"/*/manifest.json 2>/dev/null | head -1 || true)"
    if [[ -n "${newest}" ]]; then
        dirname "${newest}"
    fi
}

run_peen_capture_weights() {
    # Usage: run_peen_capture_weights <state_key> peen args...
    # Write peen output to the log file first (no pipe). Piping peen|tee|extract
    # can SIGPIPE peen if extract fails, leaving an empty log and no job hint.
    local state_key="$1"
    shift
    local logfile rc
    logfile="$(state_path "${state_key}.log")"
    echo "+ peen $*"
    echo "  (log: ${logfile})"

    set +e
    PYTHONUNBUFFERED=1 peen "$@" >"${logfile}" 2>&1
    rc=$?
    set -e

    if [[ "${rc}" -ne 0 ]]; then
        echo "Error: peen exited ${rc}; see ${logfile}" >&2
        if [[ -s "${logfile}" ]]; then
            echo "---- tail ${logfile} ----" >&2
            tail -n 40 "${logfile}" >&2 || true
        else
            echo "  (log empty)" >&2
        fi
        _hint_train_job_logs >&2
        exit 1
    fi

    if ! extract_weights_id "${logfile}" > "$(state_path "${state_key}.tmp")"; then
        echo "Error: peen finished but no weights_id in ${logfile}" >&2
        if [[ -s "${logfile}" ]]; then
            echo "---- tail ${logfile} ----" >&2
            tail -n 40 "${logfile}" >&2 || true
        fi
        _hint_train_job_logs >&2
        rm -f "$(state_path "${state_key}.tmp")"
        exit 1
    fi
    mv "$(state_path "${state_key}.tmp")" "$(state_path "${state_key}")"
    echo "Wrote state ${state_key}=$(cat "$(state_path "${state_key}")")"
}

_hint_train_job_logs() {
    local job_dir root
    root="${TRAINING_JOBS_ROOT:-${EXP_DIR}/../../../services/pe-ensemble/jobs}"
    # Normalize .. in path for display
    root="$(cd "${root}" 2>/dev/null && pwd || echo "${root}")"
    job_dir="$(latest_train_job_dir || true)"
    echo "  TRAINING_JOBS_ROOT: ${root}"
    if [[ -n "${job_dir}" ]]; then
        echo "  latest train job: ${job_dir}"
        echo "  try: cat ${job_dir}/manifest.json"
        echo "       tail -n 100 ${job_dir}/train.log"
    else
        echo "  no ${root}/*/manifest.json found"
        echo "  (peen may have died before creating a train job)"
    fi
}

# Author CV5 on library-diverse (folds 0–4); no outer random test holdout.
append_library_diverse_cv_args() {
    local -n _args="$1"
    _args+=(
        --split-strategy cv
        --cv-folds "${CV_FOLDS}"
        --use-original-fold
        --split-random-state "${SPLIT_RANDOM_STATE}"
    )
}

# Evaluate registered FT weights on the held-out author fold (default: 4).
evaluate_library_diverse_test_fold() {
    local weights_id="$1"
    local cell="$2"
    local fold="${3:-${LD_TEST_FOLD}}"
    local tag="${4:-eval}"
    local logfile
    logfile="$(state_path "${tag}_${cell}_fold${fold}.log")"

    local args=(
        evaluate
        --model "${MODEL}"
        --weights "${weights_id}"
        --benchmark-name "${NAME_FT_PREFIX}-${tag}-${cell}-fold${fold}"
        --custom-benchmark
        --study pridict2 --dataset library-diverse
        --cell-line "${cell}" --pe-system "${PE_SYSTEM}"
        --split-strategy holdout_2
        --use-original-fold
        --original-fold-test-value "${fold}"
        --device "${DEVICE}"
        --sync
    )
    echo "+ peen ${args[*]}"
    echo "  (log: ${logfile})"
    peen "${args[@]}" 2>&1 | tee "${logfile}"
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

print_repro_banner() {
    local title="$1"
    print_experiment_banner "${title}"
    echo "STATE_DIR: ${STATE_DIR}"
    echo "MODEL:     ${MODEL}"
    if [[ "${SMOKE:-0}" == "1" ]]; then
        echo "SMOKE:     1 (n_trials=${N_TRIALS}, cv_folds=${CV_FOLDS}; mini data unless SMOKE_FULL_DATA=1)"
    fi
    echo ""
}

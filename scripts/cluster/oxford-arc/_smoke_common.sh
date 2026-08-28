#!/usr/bin/env bash
# Shared SMOKE helpers: subsampled DATA_ROOT + isolated peen artifact dirs.
# Sourced by pridict2-reproduction/_common.sh and preflight.sh when SMOKE=1.
#
# Env (optional):
#   N_ROWS            rows per mini sheet (default 128)
#   WORK_DIR          scratch root (default /tmp/pe-hub-smoke-<pid>)
#   SOURCE_DATASETS   real datasets/ tree to sample (default <repo>/datasets)
#   KEEP_WORK=1       retain WORK_DIR (preflight sets trap; reproduction: export manually)
#   SMOKE_FULL_DATA=1 skip mini DATA_ROOT (use real DATA_ROOT; for ARC 1-trial smoke)

_smoke_arc_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

_smoke_repo_root() {
    cd "$(_smoke_arc_dir)/../../.." && pwd
}

# Build mini DATA_ROOT once and point peen artifact env vars at WORK_DIR.
setup_smoke_mini_data_root() {
    if [[ "${SMOKE_DATA_ROOT_READY:-}" == "1" ]]; then
        return 0
    fi
    if [[ "${SMOKE_FULL_DATA:-0}" == "1" ]]; then
        echo "SMOKE=1 SMOKE_FULL_DATA=1: using DATA_ROOT=${DATA_ROOT:-$(_smoke_repo_root)/datasets} (no subsample)"
        export SMOKE_DATA_ROOT_READY=1
        return 0
    fi

    local arc_dir repo_root py
    arc_dir="$(_smoke_arc_dir)"
    repo_root="$(_smoke_repo_root)"
    py="$(command -v python 2>/dev/null || command -v python3)"
    [[ -n "${py}" ]] || { echo "Error: python required for SMOKE mini DATA_ROOT" >&2; return 1; }

    export WORK_DIR="${WORK_DIR:-/tmp/pe-hub-smoke-$$}"
    export N_ROWS="${N_ROWS:-128}"
    export SOURCE_DATASETS="${SOURCE_DATASETS:-${repo_root}/datasets}"

    echo "SMOKE: building mini DATA_ROOT (${N_ROWS} rows/sheet) → ${WORK_DIR}/datasets"
    mkdir -p "${WORK_DIR}"
    "${py}" "${arc_dir}/preflight_build_mini_data.py" \
        --work-dir "${WORK_DIR}" \
        --source-datasets "${SOURCE_DATASETS}" \
        --n-rows "${N_ROWS}"

    export DATA_ROOT="${WORK_DIR}/datasets"
    export TRAINING_PRESETS_ROOT="${WORK_DIR}/presets_local"
    export TRAINING_SHIPPED_PRESETS_ROOT="${repo_root}/services/pe-ensemble/config/training_presets"
    export TUNING_STUDIES_ROOT="${WORK_DIR}/tuning_studies"
    export TRAINING_JOBS_ROOT="${WORK_DIR}/jobs"
    export TUNING_JOBS_ROOT="${WORK_DIR}/tune_jobs"
    export WEIGHTS_ROOT="${WORK_DIR}/weights"
    export STATE_DIR="${WORK_DIR}/state"
    mkdir -p \
        "${TRAINING_PRESETS_ROOT}" \
        "${TUNING_STUDIES_ROOT}" \
        "${TRAINING_JOBS_ROOT}" \
        "${TUNING_JOBS_ROOT}" \
        "${WEIGHTS_ROOT}" \
        "${STATE_DIR}"
    if [[ ! -s "${WEIGHTS_ROOT}/registry.json" ]]; then
        printf '%s\n' '{"schema_version":1,"entries":[]}' > "${WEIGHTS_ROOT}/registry.json"
    fi
    if [[ ! -s "${WEIGHTS_ROOT}/local_registry.json" ]]; then
        printf '%s\n' '{"schema_version":1,"entries":[]}' > "${WEIGHTS_ROOT}/local_registry.json"
    fi

    export SMOKE_DATA_ROOT_READY=1
    echo "SMOKE: DATA_ROOT=${DATA_ROOT}"
    echo "SMOKE: artifacts under ${WORK_DIR}"
}

# Default hyperparameters for smoke tune/train (MSEloss single head).
smoke_fixed_hp_json() {
    echo '{"num_epochs":2,"batch_size":16,"lr":0.0001,"loss_func":"MSEloss","y_ref":["averageedited"]}'
}

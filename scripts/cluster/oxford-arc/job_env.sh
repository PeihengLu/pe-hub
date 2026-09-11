#!/usr/bin/env bash
# Shared setup for Oxford ARC peen jobs (sourced by *.sbatch).
# Expects PE_HUB_ROOT, ARC_MODULES, CONDA_ENV from env.sh (or the environment).
#
# Always ``conda activate $CONDA_ENV`` *after* ``module load``. Skipping activate
# when CONDA_PREFIX already has peen is unsafe: ``--export=ALL`` copies a login
# activation, then Anaconda3 prepends the module ``python`` and scratch-benchmark
# runs that interpreter (no pe_common).

set -euo pipefail

_load_arc_modules() {
    if [[ -z "${ARC_MODULES:-}" ]]; then
        return 0
    fi
    if ! command -v module >/dev/null 2>&1; then
        # Login shells often need the modules init script.
        if [[ -f /etc/profile.d/modules.sh ]]; then
            # shellcheck disable=SC1091
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            # shellcheck disable=SC1091
            source /usr/share/Modules/init/bash
        fi
    fi
    if ! command -v module >/dev/null 2>&1; then
        echo "Error: 'module' not available; cannot load ARC_MODULES=${ARC_MODULES}" >&2
        exit 1
    fi
    # shellcheck disable=SC2086
    module load ${ARC_MODULES}
}

_conda_env_ref() {
    local env_ref="${CONDA_ENV:-${DATA:+$DATA/envs/pe-hub}}"
    : "${env_ref:?Set CONDA_ENV to your \$DATA env prefix (or named env)}"
    printf '%s\n' "${env_ref}"
}

_find_conda_sh() {
    if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "${CONDA_ROOT}/etc/profile.d/conda.sh"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
            printf '%s\n' "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "${HOME}/miniconda3/etc/profile.d/conda.sh"
        return 0
    fi
    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        printf '%s\n' "${HOME}/anaconda3/etc/profile.d/conda.sh"
        return 0
    fi
    return 1
}

# Put $1/bin first on PATH (idempotent) so ``command -v python`` cannot stay on
# the Anaconda3 module after ``module load``.
_pin_conda_prefix_bin() {
    local prefix="$1"
    local bindir="${prefix}/bin"
    local part joined
    local -a rest=()
    local -a _path_parts=()
    if [[ ! -d "${bindir}" ]]; then
        echo "Error: conda prefix bin missing: ${bindir}" >&2
        exit 1
    fi
    IFS=':' read -r -a _path_parts <<< "${PATH}"
    for part in "${_path_parts[@]}"; do
        if [[ -n "${part}" && "${part}" != "${bindir}" ]]; then
            rest+=("${part}")
        fi
    done
    if [[ ${#rest[@]} -gt 0 ]]; then
        joined="$(IFS=':'; printf '%s' "${rest[*]}")"
        PATH="${bindir}:${joined}"
    else
        PATH="${bindir}"
    fi
    export PATH
    hash -r 2>/dev/null || true
}

_realpath_or_self() {
    local path="$1"
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "${path}" 2>/dev/null || printf '%s\n' "${path}"
    else
        printf '%s\n' "${path}"
    fi
}

_require_job_python() {
    local prefix="${CONDA_PREFIX:-}"
    local expected actual
    if [[ -z "${prefix}" || ! -x "${prefix}/bin/python" ]]; then
        echo "Error: CONDA_PREFIX python missing (${prefix:-unset})." >&2
        echo "  Set CONDA_ENV in env.sh and run setup_interactive.sh on an interactive node." >&2
        exit 1
    fi
    expected="$(_realpath_or_self "${prefix}/bin/python")"
    actual="$(command -v python 2>/dev/null || true)"
    if [[ -z "${actual}" ]]; then
        echo "Error: python not on PATH after conda activate." >&2
        exit 1
    fi
    actual="$(_realpath_or_self "${actual}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Error: job python is ${actual}, expected ${expected}." >&2
        echo "  module load Anaconda3 must not win over \$CONDA_ENV." >&2
        exit 1
    fi
    export PYTHON="${prefix}/bin/python"
    if ! "${PYTHON}" -c "import pe_common" 2>/dev/null; then
        echo "Error: ${PYTHON} cannot import pe_common." >&2
        echo "  On an interactive node: cd \$PE_HUB_ROOT && ./scripts/install-clis.sh" >&2
        exit 1
    fi
}

_activate_conda() {
    local env_ref conda_sh
    env_ref="$(_conda_env_ref)"

    conda_sh="$(_find_conda_sh || true)"

    if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
        # shellcheck disable=SC1090
        source "${conda_sh}"
        # Drop login-shell activation copied by sbatch --export=ALL, then
        # activate the job prefix so PATH is rewritten after module load.
        local _lvl=0
        while [[ "${CONDA_SHLVL:-0}" -gt 0 && "${_lvl}" -lt 8 ]]; do
            conda deactivate || break
            _lvl="$((_lvl + 1))"
        done
        conda activate "${env_ref}"
    elif command -v conda >/dev/null 2>&1; then
        # ARC docs often use: source activate $DATA/myenv
        # shellcheck disable=SC1091
        source activate "${env_ref}" 2>/dev/null || conda activate "${env_ref}"
    else
        echo "Error: conda not found. Load Anaconda via ARC_MODULES or set CONDA_ROOT." >&2
        echo "  On interactive node: module spider Anaconda" >&2
        exit 1
    fi

    if [[ -z "${CONDA_PREFIX:-}" || ! -d "${CONDA_PREFIX}/bin" ]]; then
        echo "Error: conda activate ${env_ref} did not set CONDA_PREFIX." >&2
        exit 1
    fi
    _pin_conda_prefix_bin "${CONDA_PREFIX}"
    _require_job_python
}

if [[ "${JOB_ENV_SOURCE_LIB:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

: "${PE_HUB_ROOT:?Set PE_HUB_ROOT (repo checkout under \$DATA)}"

_load_arc_modules
_activate_conda

cd "${PE_HUB_ROOT}"
export DATA_ROOT="${DATA_ROOT:-${PE_HUB_ROOT}/datasets}"
export DEVICE="${DEVICE:-cuda:0}"

# Prefer CUDA when a GPU was allocated.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${DEVICE}" == "auto" ]]; then
    export DEVICE="cuda:0"
fi

echo "==== ARC job env ===="
echo "host:       $(hostname)"
echo "job:        ${SLURM_JOB_ID:-local}"
echo "PE_HUB:     ${PE_HUB_ROOT}"
echo "DATA_ROOT:  ${DATA_ROOT}"
echo "DEVICE:     ${DEVICE}"
echo "modules:    ${ARC_MODULES:-none}"
echo "conda:      ${CONDA_PREFIX:-unset}"
echo "python:     $(command -v python 2>/dev/null || echo missing)"
echo "PYTHON:     ${PYTHON:-unset}"
echo "CUDA_VIS:   ${CUDA_VISIBLE_DEVICES:-unset}"
command -v peen >/dev/null || {
    echo "Error: peen not on PATH. On an interactive node: cd \$PE_HUB_ROOT && ./scripts/install-clis.sh" >&2
    exit 1
}
peen devices || true
echo "====================="

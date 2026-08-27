#!/usr/bin/env bash
# Shared setup for Oxford ARC peen jobs (sourced by *.sbatch).
# Expects PE_HUB_ROOT, ARC_MODULES, CONDA_ENV from env.sh (or the environment).

set -euo pipefail

: "${PE_HUB_ROOT:?Set PE_HUB_ROOT (repo checkout under \$DATA)}"

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

_activate_conda() {
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/peen" ]]; then
        return 0
    fi

    local conda_sh=""
    if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        conda_sh="${CONDA_ROOT}/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # After module load Anaconda3/Mamba, conda is on PATH.
        conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
    fi

    local env_ref="${CONDA_ENV:-${DATA:+$DATA/envs/pe-hub}}"
    : "${env_ref:?Set CONDA_ENV to your \$DATA env prefix (or named env)}"

    if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
        # shellcheck disable=SC1090
        source "${conda_sh}"
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
}

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
echo "CUDA_VIS:   ${CUDA_VISIBLE_DEVICES:-unset}"
command -v peen >/dev/null || {
    echo "Error: peen not on PATH. On an interactive node: cd \$PE_HUB_ROOT && ./scripts/install-clis.sh" >&2
    exit 1
}
peen devices || true
echo "====================="

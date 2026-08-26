#!/usr/bin/env bash
# Shared setup for Oxford ARC peen jobs (sourced by *.sbatch).
# Expects PE_HUB_ROOT and CONDA_* from env.sh (or the environment).

set -euo pipefail

: "${PE_HUB_ROOT:?Set PE_HUB_ROOT (repo checkout under \$DATA)}"

if [[ -n "${ARC_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load ${ARC_MODULES}
fi

_activate_conda() {
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/peen" ]]; then
        return 0
    fi
    local conda_sh=""
    if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        conda_sh="${CONDA_ROOT}/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
        echo "Error: conda.sh not found. Set CONDA_ROOT in env.sh" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${CONDA_ENV:-pe-hub}"
}

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
echo "conda:      ${CONDA_PREFIX:-unset}"
echo "CUDA_VIS:   ${CUDA_VISIBLE_DEVICES:-unset}"
command -v peen >/dev/null || {
    echo "Error: peen not on PATH. On an interactive node: cd \$PE_HUB_ROOT && ./scripts/install-clis.sh" >&2
    exit 1
}
peen devices || true
echo "====================="

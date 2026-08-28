#!/usr/bin/env bash
# Oxford ARC (htc) SLURM job: run a pe-hub experiment/stage script.
#
# Prefer submit.sh (sets paths + SBATCH overrides). Direct use:
#   sbatch --export=ALL,STAGE_SCRIPT=01_tune_base_library1.sh \
#     scripts/cluster/oxford-arc/run_stage.sbatch
#
# Or pass a relative path under scripts/experiments/pridict2-reproduction/:
#   STAGE_SCRIPT=03_train_base_library1.sh
#
# Or an absolute path / any executable:
#   STAGE_SCRIPT=/path/to/custom.sh

#SBATCH --job-name=pe-hub
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Slurm copies this script to /var/spool/slurmd/... — do not resolve paths via BASH_SOURCE.
if [[ -n "${PE_HUB_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/cluster/oxford-arc/job_env.sh" ]]; then
    PE_HUB_ROOT="${SLURM_SUBMIT_DIR}"
else
    _arc_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${_arc_self}/job_env.sh" ]]; then
        ARC_DIR="${_arc_self}"
        PE_HUB_ROOT="$(cd "${ARC_DIR}/../../.." && pwd)"
    fi
fi
: "${PE_HUB_ROOT:?Set PE_HUB_ROOT (submit.sh sources env.sh and exports it)}"
ARC_DIR="${PE_HUB_ROOT}/scripts/cluster/oxford-arc"

if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi
# shellcheck source=./job_env.sh
source "${ARC_DIR}/job_env.sh"

STAGE_SCRIPT="${STAGE_SCRIPT:?Set STAGE_SCRIPT (e.g. 01_tune_base_library1.sh)}"
REPRO_DIR="${PE_HUB_ROOT}/scripts/experiments/pridict2-reproduction"

if [[ "${STAGE_SCRIPT}" != /* ]]; then
    if [[ -x "${REPRO_DIR}/${STAGE_SCRIPT}" ]]; then
        STAGE_SCRIPT="${REPRO_DIR}/${STAGE_SCRIPT}"
    elif [[ -x "${PE_HUB_ROOT}/${STAGE_SCRIPT}" ]]; then
        STAGE_SCRIPT="${PE_HUB_ROOT}/${STAGE_SCRIPT}"
    fi
fi

if [[ ! -x "${STAGE_SCRIPT}" && ! -f "${STAGE_SCRIPT}" ]]; then
    echo "Error: STAGE_SCRIPT not found: ${STAGE_SCRIPT}" >&2
    exit 1
fi

chmod +x "${STAGE_SCRIPT}" 2>/dev/null || true

echo "+ ${STAGE_SCRIPT}"
# shellcheck disable=SC2086
exec bash "${STAGE_SCRIPT}" ${STAGE_EXTRA_ARGS:-}

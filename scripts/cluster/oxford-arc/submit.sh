#!/usr/bin/env bash
# Submit a pe-hub stage to Oxford ARC HTC (GPU).
#
# Usage (from anywhere):
#   cd /path/to/pe-hub
#   cp scripts/cluster/oxford-arc/env.sh.example scripts/cluster/oxford-arc/env.sh
#   # edit env.sh
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
#   ./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh
#
# Smoke / overrides:
#   SMOKE=1 ARC_PARTITION=short ARC_TIME=01:00:00 \
#     ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
#
# Extra sbatch flags after -- :
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh -- --constraint='gpu_sku:A100'
#
# Dry-run (validate + estimated start):
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
else
    echo "Warning: ${ARC_DIR}/env.sh missing — using env.sh.example defaults / current env" >&2
    # shellcheck source=./env.sh.example
    source "${ARC_DIR}/env.sh.example"
fi

STAGE_SCRIPT="${1:?Usage: $0 <stage_script.sh> [-- extra sbatch args]}"
shift || true

EXTRA_SBATCH=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_SBATCH=("$@")
fi

SBATCH_SCRIPT="${ARC_DIR}/run_stage.sbatch"
JOB_NAME="pe-$(basename "${STAGE_SCRIPT}" .sh | tr '_' '-' | cut -c1-40)"

SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --clusters="${ARC_CLUSTER}"
    --partition="${ARC_PARTITION}"
    --time="${ARC_TIME}"
    --cpus-per-task="${ARC_CPUS}"
    --mem="${ARC_MEM}"
    --gres="gpu:${ARC_GPUS}"
    --export=ALL,PE_HUB_ROOT="${PE_HUB_ROOT}",STAGE_SCRIPT="${STAGE_SCRIPT}"
    --chdir="${PE_HUB_ROOT}"
)

if [[ -n "${ARC_GPU_CONSTRAINT:-}" ]]; then
    SBATCH_ARGS+=(--constraint="${ARC_GPU_CONSTRAINT}")
fi
if [[ -n "${ARC_MAIL_USER:-}" ]]; then
    SBATCH_ARGS+=(--mail-type=END,FAIL --mail-user="${ARC_MAIL_USER}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "+ sbatch --test-only ${SBATCH_ARGS[*]} ${EXTRA_SBATCH[*]} ${SBATCH_SCRIPT}"
    sbatch --test-only "${SBATCH_ARGS[@]}" "${EXTRA_SBATCH[@]}" "${SBATCH_SCRIPT}"
    exit 0
fi

echo "Submitting ${STAGE_SCRIPT}"
echo "  cluster=${ARC_CLUSTER} partition=${ARC_PARTITION} time=${ARC_TIME} gpus=${ARC_GPUS}"
echo "  PE_HUB_ROOT=${PE_HUB_ROOT} DEVICE=${DEVICE}"
sbatch "${SBATCH_ARGS[@]}" "${EXTRA_SBATCH[@]}" "${SBATCH_SCRIPT}"

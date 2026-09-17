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
#   SMOKE=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
#     → 1 trial, 2-fold CV on full data (submit.sh sets SMOKE_FULL_DATA=1)
#   SMOKE=1 ARC_PARTITION=short ARC_TIME=01:00:00 \
#     ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
#
# Local smoke (mini DATA_ROOT, same stage script):
#   SMOKE=1 ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh
#   SMOKE=1 ./scripts/cluster/oxford-arc/preflight.sh
#
# Extra sbatch flags after -- :
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh -- --constraint='gpu_sku:L40S'
#
# SLURM dependency (e.g. from submit_arc_pipeline.sh):
#   ARC_DEPENDENCY=afterok:123456 ./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh
#
# Dry-run (validate + estimated start):
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
#
# Scratch-benchmark matrix: default is one short L40S job per seed
# (7 datasets × 3 models × 3 seeds = 63). Set SUBMIT_MATRIX_AS_ONE=1 to pack
# every cell into a single job. Set INDEX + MODEL + BENCHMARK for one seed.
#   ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   MODEL=oped BENCHMARK=pridict1-library1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   SUBMIT_MATRIX_AS_ONE=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
#   RUN_ID=<id> ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preserve caller SLURM/device overrides across env.sh. Some checkouts assign
# ARC_PARTITION/ARC_TIME unconditionally (without ${VAR:-default}), which would
# otherwise ignore prefix env on: ARC_TIME=24:00:00 ./submit.sh …
_SAVE_ARC_KEYS=(
    ARC_CLUSTER ARC_PARTITION ARC_TIME ARC_GPUS ARC_GPU_CONSTRAINT
    ARC_CPUS ARC_MEM ARC_MAIL_USER DEVICE
)
for _k in "${_SAVE_ARC_KEYS[@]}"; do
    if [[ -n "${!_k+x}" ]]; then
        printf -v "_SAVE_${_k}" '%s' "${!_k}"
    else
        unset "_SAVE_${_k}" 2>/dev/null || true
    fi
done

if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
else
    echo "Warning: ${ARC_DIR}/env.sh missing — using env.sh.example defaults / current env" >&2
    # shellcheck source=./env.sh.example
    source "${ARC_DIR}/env.sh.example"
fi

for _k in "${_SAVE_ARC_KEYS[@]}"; do
    _sk="_SAVE_${_k}"
    if [[ -n "${!_sk+x}" ]]; then
        export "${_k}=${!_sk}"
    fi
done
unset _k _sk _SAVE_ARC_KEYS
for _k in ARC_CLUSTER ARC_PARTITION ARC_TIME ARC_GPUS ARC_GPU_CONSTRAINT ARC_CPUS ARC_MEM ARC_MAIL_USER DEVICE; do
    unset "_SAVE_${_k}" 2>/dev/null || true
done
unset _k

STAGE_SCRIPT="${1:?Usage: $0 <stage_script.sh> [-- extra sbatch args]}"
shift || true

EXTRA_SBATCH=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_SBATCH=("$@")
fi

STAGE_BASENAME="$(basename "${STAGE_SCRIPT}")"

# Matrix stages 01/02 default to one job per seed via submit_arc_matrix.sh.
# Skip the fan-out when already inside that helper, when packing all cells
# (SUBMIT_MATRIX_AS_ONE=1), or when the caller pinned a single INDEX.
# MODEL=oped BENCHMARK=… submit.sh 01_tune_matrix.sh  → 3 seed jobs
# INDEX=0 MODEL=oped BENCHMARK=… submit.sh 01_tune_matrix.sh  → one seed
# SUBMIT_MATRIX_AS_ONE=1 submit.sh 01_tune_matrix.sh  → one job, all cells
if [[ "${STAGE_BASENAME}" == 01_tune_matrix.sh || "${STAGE_BASENAME}" == 02_train_matrix.sh ]] \
    && [[ "${SUBMIT_MATRIX_AS_ONE:-0}" != "1" ]] \
    && [[ "${_ARC_MATRIX_CELL:-0}" != "1" ]] \
    && [[ -z "${INDEX:-}" ]]; then
    MATRIX_SUBMIT="${PE_HUB_ROOT}/scripts/experiments/scratch-benchmark/submit_arc_matrix.sh"
    if [[ -x "${MATRIX_SUBMIT}" ]]; then
        echo "Matrix stage ${STAGE_BASENAME}: submitting one SLURM job per seed (set SUBMIT_SEEDS=0 for one job per cell, SUBMIT_MATRIX_AS_ONE=1 for all cells in one job)."
        exec "${MATRIX_SUBMIT}" "${STAGE_BASENAME}"
    fi
    echo "Warning: ${MATRIX_SUBMIT} missing; submitting ${STAGE_BASENAME} as a single job." >&2
fi

SBATCH_SCRIPT="${ARC_DIR}/run_stage.sbatch"
JOB_NAME="pe-$(basename "${STAGE_SCRIPT}" .sh | tr '_' '-' | cut -c1-40)"
if [[ -n "${MODEL:-}" && -n "${BENCHMARK:-}" ]]; then
    JOB_SUFFIX="$(echo "${MODEL}__${BENCHMARK}" | tr '[:upper:]' '[:lower:]' | tr '_/' '-' | tr -cd 'a-z0-9.-')"
    JOB_NAME="pe-$(basename "${STAGE_SCRIPT}" .sh | tr '_' '-' | cut -c1-24)-${JOB_SUFFIX}"
    if [[ -n "${INDEX:-}" ]]; then
        JOB_NAME="${JOB_NAME}-s${INDEX}"
    fi
    JOB_NAME="${JOB_NAME:0:40}"
fi

# On cluster, SMOKE=1 means fewer trials — not subsampled data (local default).
if [[ "${SMOKE:-0}" == "1" && -z "${SMOKE_FULL_DATA:-}" ]]; then
    export SMOKE_FULL_DATA=1
fi

SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --clusters="${ARC_CLUSTER}"
    --partition="${ARC_PARTITION}"
    --time="${ARC_TIME}"
    --cpus-per-task="${ARC_CPUS}"
    --mem="${ARC_MEM}"
    --gres="gpu:${ARC_GPUS}"
    --parsable
    --export=ALL,PE_HUB_ROOT="${PE_HUB_ROOT}",STAGE_SCRIPT="${STAGE_SCRIPT}",SMOKE="${SMOKE:-0}",SMOKE_FULL_DATA="${SMOKE_FULL_DATA:-0}",MODEL="${MODEL:-}",BENCHMARK="${BENCHMARK:-}",RUN_ID="${RUN_ID:-}",N_SEEDS="${N_SEEDS:-3}",N_TRIALS="${N_TRIALS:-10}",PROTOCOL="${PROTOCOL:-holdout_3}",INDEX="${INDEX:-}",SKIP_IF_DONE="${SKIP_IF_DONE:-0}",SKIP_IF_TUNED="${SKIP_IF_TUNED:-0}",PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}",CELL_LINE="${CELL_LINE:-}",FOLD="${FOLD:-}",FIXED_HP_JSON="${FIXED_HP_JSON:-}"
    --chdir="${PE_HUB_ROOT}"
)

if [[ -n "${ARC_GPU_CONSTRAINT:-}" ]]; then
    SBATCH_ARGS+=(--constraint="${ARC_GPU_CONSTRAINT}")
fi
if [[ -n "${ARC_MAIL_USER:-}" ]]; then
    SBATCH_ARGS+=(--mail-type=END,FAIL --mail-user="${ARC_MAIL_USER}")
fi
if [[ -n "${ARC_DEPENDENCY:-}" ]]; then
    SBATCH_ARGS+=(--dependency="${ARC_DEPENDENCY}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "+ sbatch --test-only ${SBATCH_ARGS[*]} ${EXTRA_SBATCH[*]} ${SBATCH_SCRIPT}"
    sbatch --test-only "${SBATCH_ARGS[@]}" "${EXTRA_SBATCH[@]}" "${SBATCH_SCRIPT}"
    exit 0
fi

echo "Submitting ${STAGE_SCRIPT}"
echo "  cluster=${ARC_CLUSTER} partition=${ARC_PARTITION} time=${ARC_TIME} gpus=${ARC_GPUS}"
echo "  PE_HUB_ROOT=${PE_HUB_ROOT} DEVICE=${DEVICE} SMOKE=${SMOKE:-0}"
if [[ -n "${ARC_DEPENDENCY:-}" ]]; then
    echo "  dependency=${ARC_DEPENDENCY}"
fi
_SBATCH_OUT="$(sbatch "${SBATCH_ARGS[@]}" "${EXTRA_SBATCH[@]}" "${SBATCH_SCRIPT}")"
# --parsable → "jobid" or "jobid;cluster"
SUBMIT_JOB_ID="${_SBATCH_OUT%%;*}"
export SUBMIT_JOB_ID
echo "Submitted batch job ${SUBMIT_JOB_ID}${_SBATCH_OUT#${SUBMIT_JOB_ID}}"
echo "${SUBMIT_JOB_ID}"

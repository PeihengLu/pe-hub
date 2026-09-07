#!/usr/bin/env bash
# Datasheet / dataset from-scratch benchmark (nested Optuna).
#
# Usage:
#   ./scripts/experiments/datasheet-benchmark/run.sh \
#     --model pridict2 --n 5 -x 20 \
#     --study minsepie --dataset library-insert-set12 \
#     --cell-line hek293t --pe-system pe2 \
#     --edit-type ins
#
# Env: DEVICE, N, N_TRIALS, SMOKE, SIZE_THRESHOLD, PROTOCOL, BASE_SEED
# Extra args after the script name are forwarded to run_benchmark.py.
# If --n / --n-trials are omitted, N and N_TRIALS (or CV_FOLDS) env vars are used.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_DIR="$(cd "${SCRIPT_DIR}/../../hyperparameter" && pwd)"
# shellcheck source=../../hyperparameter/_common.sh
source "${HP_DIR}/_common.sh"
require_peen

if [[ "${SMOKE:-0}" == "1" ]]; then
    ARC_SMOKE_DIR="$(cd "${SCRIPT_DIR}/../../cluster/oxford-arc" && pwd)"
    if [[ -f "${ARC_SMOKE_DIR}/_smoke_common.sh" ]]; then
        # shellcheck source=../../cluster/oxford-arc/_smoke_common.sh
        source "${ARC_SMOKE_DIR}/_smoke_common.sh"
        setup_smoke_mini_data_root
    fi
fi

N_VALUE="${N:-${CV_FOLDS:-}}"
X_VALUE="${N_TRIALS:-}"

FORWARD=()
HAVE_N=0
HAVE_X=0
for arg in "$@"; do
    case "${arg}" in
        --n|--n=*) HAVE_N=1 ;;
        -x|--n-trials|--n-trials=*) HAVE_X=1 ;;
    esac
done

if [[ "${HAVE_N}" -eq 0 ]]; then
    if [[ -z "${N_VALUE}" ]]; then
        echo "Error: pass --n or set N (folds / random seeds)." >&2
        exit 1
    fi
    FORWARD+=(--n "${N_VALUE}")
fi
if [[ "${HAVE_X}" -eq 0 ]]; then
    if [[ -z "${X_VALUE}" ]]; then
        echo "Error: pass --n-trials / -x or set N_TRIALS." >&2
        exit 1
    fi
    FORWARD+=(--n-trials "${X_VALUE}")
fi

if [[ -n "${DEVICE:-}" ]]; then
    FORWARD+=(--device "${DEVICE}")
fi
if [[ -n "${PROTOCOL:-}" ]]; then
    FORWARD+=(--protocol "${PROTOCOL}")
fi
if [[ -n "${SIZE_THRESHOLD:-}" ]]; then
    FORWARD+=(--size-threshold "${SIZE_THRESHOLD}")
fi
if [[ -n "${SPLIT_RANDOM_STATE:-}" ]]; then
    FORWARD+=(--base-seed "${SPLIT_RANDOM_STATE}")
fi

if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON:-}" ]]; then
    _smoke_arc="$(cd "${SCRIPT_DIR}/../../cluster/oxford-arc" && pwd)"
    if [[ -f "${_smoke_arc}/_smoke_common.sh" ]]; then
        # shellcheck source=../../cluster/oxford-arc/_smoke_common.sh
        source "${_smoke_arc}/_smoke_common.sh"
        FIXED_HP_JSON="$(smoke_fixed_hp_json)"
    fi
fi
if [[ -n "${FIXED_HP_JSON:-}" ]]; then
    FORWARD+=(--fixed-hyperparameters-json "${FIXED_HP_JSON}")
fi
print_experiment_banner "Datasheet benchmark (nested Optuna)"
echo "protocol env: ${PROTOCOL:-auto}"
echo "N (folds/seeds): CLI or ${N_VALUE:-unset}"
echo "X (trials):      CLI or ${X_VALUE:-unset}"
echo ""

exec python "${SCRIPT_DIR}/run_benchmark.py" "${FORWARD[@]}" "$@"

#!/usr/bin/env bash
# Generic Optuna HPO with holdout_3 (70/15/15 train/val/test).
# Optuna optimizes on the validation partition; test is held out until evaluate.
#
# Usage:
#   ./scripts/hyperparameter/tune_hpo_holdout3.sh --model oped \
#     --dataset-name my-run --study pridict1 --dataset library1
#
# Env: N_TRIALS, DEVICE, TRAIN_PCT, VAL_PCT, TEST_PCT, SPLIT_RANDOM_STATE, SMOKE,
#      FIXED_HP_JSON, STUDY_NAME, SKIP_IF_TUNED, DATASET_KEY

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 --model <name> --dataset-name <label> [--study ...] [--dataset ...] ..." >&2
    exit 1
fi

TRAIN_PCT="${TRAIN_PCT:-0.7}"
VAL_PCT="${VAL_PCT:-0.15}"
TEST_PCT="${TEST_PCT:-0.15}"

echo "======================================"
echo "HPO: holdout_3 (train/val/test)"
echo "======================================"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "n_trials:  ${N_TRIALS}"
echo "device:    ${DEVICE}"
echo "train_pct: ${TRAIN_PCT}  val_pct: ${VAL_PCT}  test_pct: ${TEST_PCT}"
echo "split_random_state: ${SPLIT_RANDOM_STATE}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "SMOKE:     1 (mini data unless SMOKE_FULL_DATA=1)"
fi
echo ""

EXTRA_ARGS=("$@")
STUDY_NAME="${STUDY_NAME:-}"
FIXED_HP_JSON="${FIXED_HP_JSON:-}"
DATASET_KEY="${DATASET_KEY:-}"

if [[ -z "${DATASET_KEY}" ]]; then
    _study=""; _dataset=""; _cell=""; _pe=""; _model=""
    _args=("${EXTRA_ARGS[@]}")
    for ((i = 0; i < ${#_args[@]}; i++)); do
        case "${_args[$i]}" in
            --model) _model="${_args[$((i + 1))]:-}" ;;
            --study) _study="${_args[$((i + 1))]:-}" ;;
            --dataset) _dataset="${_args[$((i + 1))]:-}" ;;
            --cell-line) _cell="${_args[$((i + 1))]:-}" ;;
            --pe-system) _pe="${_args[$((i + 1))]:-}" ;;
        esac
    done
    if [[ -n "${_study}" && -n "${_dataset}" ]]; then
        DATASET_KEY="$(echo "${_study}/${_dataset}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
        if [[ -n "${_cell}" && -n "${_pe}" ]]; then
            DATASET_KEY="${DATASET_KEY}/$(echo "${_cell}/${_pe}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
        fi
    fi
    if [[ -n "${_model}" && -n "${DATASET_KEY}" ]]; then
        maybe_skip_if_tuned "${_model}" "${DATASET_KEY}"
    fi
elif [[ -n "${DATASET_KEY}" ]]; then
    _model=""
    for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
        if [[ "${EXTRA_ARGS[$i]}" == "--model" ]]; then
            _model="${EXTRA_ARGS[$((i + 1))]:-}"
            break
        fi
    done
    if [[ -n "${_model}" ]]; then
        maybe_skip_if_tuned "${_model}" "${DATASET_KEY}"
    fi
fi

TUNE_ARGS=(
    tune
    --split-strategy holdout_3
    --train-pct "${TRAIN_PCT}"
    --val-pct "${VAL_PCT}"
    --test-pct "${TEST_PCT}"
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --n-trials "${N_TRIALS}"
    --device "${DEVICE}"
    --notes "experiment: holdout_3 HPO (train=${TRAIN_PCT} val=${VAL_PCT} test=${TEST_PCT})"
)

if [[ -n "${STUDY_NAME}" ]]; then
    TUNE_ARGS+=(--study-name "${STUDY_NAME}")
fi
if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON}" ]]; then
    _smoke_arc="$(cd "${SCRIPT_DIR}/../cluster/oxford-arc" && pwd)"
    if [[ -f "${_smoke_arc}/_smoke_common.sh" ]]; then
        # shellcheck source=../cluster/oxford-arc/_smoke_common.sh
        source "${_smoke_arc}/_smoke_common.sh"
        FIXED_HP_JSON="$(smoke_fixed_hp_json)"
    else
        FIXED_HP_JSON='{"num_epochs":2,"batch_size":16,"lr":0.0001,"loss_func":"MSEloss","y_ref":["averageedited"]}'
    fi
fi
if [[ -n "${FIXED_HP_JSON}" ]]; then
    TUNE_ARGS+=(--fixed-hyperparameters-json "${FIXED_HP_JSON}")
fi

TUNE_ARGS+=("${EXTRA_ARGS[@]}")

echo "+ peen ${TUNE_ARGS[*]}"
echo ""
exec peen "${TUNE_ARGS[@]}"

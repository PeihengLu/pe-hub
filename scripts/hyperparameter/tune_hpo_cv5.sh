#!/usr/bin/env bash
# Generic Optuna HPO: outer test holdout + 5-fold CV (random group splits).
#
# Usage:
#   ./scripts/hyperparameter/tune_hpo_cv5.sh --model pridict2 \
#     --dataset-name minsepie-set12 \
#     --study minsepie --dataset library-insert-set12 \
#     --cell-line hek293t --pe-system pe2
#
# Env: N_TRIALS, DEVICE, CV_FOLDS, TEST_PCT, SMOKE, FIXED_HP_JSON, STUDY_NAME
#      SKIP_IF_TUNED=1  # exit early if a dataset preset already exists (needs DATASET_KEY)
#      DATASET_KEY      # e.g. minsepie/library_insert_set12/hek293t/pe2

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 --model <name> --dataset-name <label> [--study ...] [--dataset ...] ..." >&2
    exit 1
fi

print_experiment_banner "HPO: 5-fold CV + outer test (random splits)"

EXTRA_ARGS=("$@")
STUDY_NAME="${STUDY_NAME:-}"
FIXED_HP_JSON="${FIXED_HP_JSON:-}"
DATASET_KEY="${DATASET_KEY:-}"

# Best-effort: derive DATASET_KEY from --study/--dataset/--cell-line/--pe-system if unset.
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
    --split-strategy cv
    --cv-folds "${CV_FOLDS}"
    --test-pct "${TEST_PCT}"
    --split-random-state "${SPLIT_RANDOM_STATE}"
    --n-trials "${N_TRIALS}"
    --device "${DEVICE}"
    --register-best-weights
    --notes "experiment: ${CV_FOLDS}-fold CV + test_pct=${TEST_PCT} HPO"
)

if [[ -n "${STUDY_NAME}" ]]; then
    TUNE_ARGS+=(--study-name "${STUDY_NAME}")
fi
if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON='{"num_epochs":3,"epochs":3,"epoch_num":3,"batch_size":64}'
fi
if [[ -n "${FIXED_HP_JSON}" ]]; then
    TUNE_ARGS+=(--fixed-hyperparameters-json "${FIXED_HP_JSON}")
fi

TUNE_ARGS+=("${EXTRA_ARGS[@]}")

echo "+ peen ${TUNE_ARGS[*]}"
echo ""
exec peen "${TUNE_ARGS[@]}"

#!/usr/bin/env bash
# Lightweight from-scratch train probe for DeepPrime, OPED, and PRIDICT2.
# Runs sequential peen train jobs and prints each job's full train.log.
#
# Datasets: pridict1/library1, pridict2/library-diverse, deepprime/deepprime-clinvar
#
# Usage:
#   conda activate pedb
#   DEVICE=mps ./scripts/experiments/probe_scratch_train.sh
#
# Env:
#   DEVICE, SMOKE=1 (first model × first dataset, 2 epochs)
#   MODELS="oped deepprime"  DATASET_NAMES="pridict1-library1"
#   NUM_WORKERS=15 (DataLoader prefetch workers; default 15)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../hyperparameter/_common.sh
source "${REPO_ROOT}/scripts/hyperparameter/_common.sh"
require_peen

PEEN_CMD=(python -m pe_ensemble.cli)
DEVICE="${DEVICE:-auto}"
SPLIT_RANDOM_STATE="${SPLIT_RANDOM_STATE:-42}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS_DEEPPRIME="${EPOCHS_DEEPPRIME:-2}"
  EPOCHS_OPED="${EPOCHS_OPED:-2}"
  EPOCHS_PRIDICT2="${EPOCHS_PRIDICT2:-2}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
else
  EPOCHS_DEEPPRIME="${EPOCHS_DEEPPRIME:-30}"
  EPOCHS_OPED="${EPOCHS_OPED:-50}"
  EPOCHS_PRIDICT2="${EPOCHS_PRIDICT2:-30}"
  BATCH_SIZE="${BATCH_SIZE:-128}"
fi
NUM_WORKERS="${NUM_WORKERS:-15}"

MODELS_ARR=(deepprime oped pridict2)
if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS_ARR=(${MODELS})
fi

# name study dataset use_original_fold
DATASETS_ALL=(
  "pridict1-library1 pridict1 library1 0"
  "pridict2-library-diverse pridict2 library-diverse 0"
  "deepprime-clinvar deepprime deepprime-clinvar 1"
)

DATASETS=()
if [[ -n "${DATASET_NAMES:-}" ]]; then
  for want in ${DATASET_NAMES}; do
    matched=0
    for row in "${DATASETS_ALL[@]}"; do
      # shellcheck disable=SC2086
      set -- ${row}
      if [[ "$1" == "${want}" ]]; then
        DATASETS+=("${row}")
        matched=1
        break
      fi
    done
    if [[ "${matched}" -ne 1 ]]; then
      echo "Unknown dataset '${want}' (pridict1-library1 | pridict2-library-diverse | deepprime-clinvar)" >&2
      exit 1
    fi
  done
else
  DATASETS=("${DATASETS_ALL[@]}")
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODELS_ARR=("${MODELS_ARR[0]}")
  DATASETS=("${DATASETS[0]}")
fi

hp_json() {
  local model="$1"
  MODEL="${model}" \
  EPOCHS_DEEPPRIME="${EPOCHS_DEEPPRIME}" \
  EPOCHS_OPED="${EPOCHS_OPED}" \
  EPOCHS_PRIDICT2="${EPOCHS_PRIDICT2}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  python - <<'PY'
import json, os
model = os.environ["MODEL"].strip().lower()
batch = int(os.environ["BATCH_SIZE"])
workers = int(os.environ["NUM_WORKERS"])
hp = {
    "load_pretrained": False,
    "batch_size": batch,
    "freezing": False,
    "num_workers": workers,
}
if model == "deepprime":
    n = int(os.environ["EPOCHS_DEEPPRIME"])
    hp.update({"epochs": n, "early_stopping_patience": n})
elif model == "oped":
    n = int(os.environ["EPOCHS_OPED"])
    hp.update({"epoch_num": n, "early_stopping_patience": n})
elif model == "pridict2":
    n = int(os.environ["EPOCHS_PRIDICT2"])
    hp.update({
        "num_epochs": n,
        "early_stopping_patience": n,
        "loss_func": "MSEloss",
        "y_ref": ["averageedited"],
    })
else:
    raise SystemExit(f"unsupported model: {model}")
print(json.dumps(hp, separators=(",", ":")))
PY
}

print_experiment_banner "From-scratch train probe"
echo "DEVICE: ${DEVICE}"
echo "num_workers: ${NUM_WORKERS}"
echo "models: ${MODELS_ARR[*]}"
echo "datasets:"
for row in "${DATASETS[@]}"; do
  echo "  ${row}"
done
echo ""

for MODEL in "${MODELS_ARR[@]}"; do
  for row in "${DATASETS[@]}"; do
    # shellcheck disable=SC2086
    set -- ${row}
    BENCH="$1"
    STUDY="$2"
    DATASET="$3"
    USE_ORIG="$4"
    DATASET_NAME="scratch-probe__${MODEL}__${BENCH}"
    HP="$(hp_json "${MODEL}")"

    echo "======================================"
    echo "TRAIN ${MODEL} @ ${BENCH}"
    echo "======================================"

    ARGS=(
      train
      --model "${MODEL}"
      --dataset-name "${DATASET_NAME}"
      --study "${STUDY}"
      --dataset "${DATASET}"
      --split-strategy holdout_3
      --train-pct 0.7 --val-pct 0.15 --test-pct 0.15
      --split-random-state "${SPLIT_RANDOM_STATE}"
      --hyperparameters-json "${HP}"
      --device "${DEVICE}"
      --notes "scratch-train-probe: ${MODEL} on ${BENCH}"
    )
    if [[ "${USE_ORIG}" == "1" ]]; then
      ARGS+=(--use-original-fold)
    else
      ARGS+=(--no-use-original-fold)
    fi

    TMP="$(mktemp)"
    set +e
    "${PEEN_CMD[@]}" "${ARGS[@]}" 2>&1 | tee "${TMP}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    JOB_ID="$(awk -F= '/^job_id=/{print $2; exit}' "${TMP}" || true)"
    JOBS_ROOT="$(awk -F= '/^jobs_root=/{print $2; exit}' "${TMP}" || true)"
    rm -f "${TMP}"
  
    if [[ -n "${JOB_ID}" && -n "${JOBS_ROOT}" && -f "${JOBS_ROOT}/${JOB_ID}/train.log" ]]; then
      echo ""
      echo "----- full train.log (${JOB_ID}) -----"
      cat "${JOBS_ROOT}/${JOB_ID}/train.log"
      echo "----- end train.log -----"
    else
      echo "(no train.log found for this run)" >&2
    fi

    if [[ "${EXIT_CODE}" -ne 0 ]]; then
      echo "Training failed with exit ${EXIT_CODE}" >&2
      exit "${EXIT_CODE}"
    fi
    echo ""
  done
done

echo "Done."

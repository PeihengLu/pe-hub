#!/usr/bin/env bash
# Shared helpers for pe-hub HPO scripts (scripts/hyperparameter/).
#
# Default protocol:
#   - Outer test holdout (--test-pct, default 0.15)
#   - 5-fold CV on the remainder for Optuna HPO
#   - DeepPrime-only exception: --use-original-fold (author folds; -1 = test)
#   - Merged DeepPrime ClinVar + PRIDICT library1: --merge --use-original-fold
#     (library1 loci that overlap ClinVar inherit DeepPrime original_fold)

set -euo pipefail

HYPERPARAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HYPERPARAM_DIR}/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"

require_peen() {
    if ! command -v peen >/dev/null 2>&1; then
        echo "Error: peen not on PATH. conda activate <env> && ./scripts/install-clis.sh" >&2
        exit 1
    fi
}

# Defaults shared by HPO runners (override via env).
N_TRIALS="${N_TRIALS:-20}"
DEVICE="${DEVICE:-auto}"
CV_FOLDS="${CV_FOLDS:-5}"
TEST_PCT="${TEST_PCT:-0.15}"
SMOKE="${SMOKE:-0}"
SPLIT_RANDOM_STATE="${SPLIT_RANDOM_STATE:-42}"

if [[ "${SMOKE}" == "1" ]]; then
    N_TRIALS="${N_TRIALS_SMOKE:-2}"
fi

print_experiment_banner() {
    local title="$1"
    echo "======================================"
    echo "${title}"
    echo "======================================"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "n_trials:  ${N_TRIALS}"
    echo "device:    ${DEVICE}"
    echo "cv_folds:  ${CV_FOLDS}"
    echo "test_pct:  ${TEST_PCT}  (ignored when --use-original-fold uses author -1)"
    echo ""
}

# Paths where peen tune persists results (local overlay, gitignored).
preset_file_for_model() {
    local model_lc
    model_lc="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    echo "${TRAINING_PRESETS_ROOT:-${REPO_ROOT}/services/pe-ensemble/config/training_presets_local}/${model_lc}.yaml"
}

shipped_preset_file_for_model() {
    local model_lc
    model_lc="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    echo "${TRAINING_SHIPPED_PRESETS_ROOT:-${REPO_ROOT}/services/pe-ensemble/config/training_presets}/${model_lc}.yaml"
}

# Return 0 if YAML has a datasets.<key> entry (exact or study/dataset parent).
# Checks local overlay first, then shipped.
preset_has_dataset_key() {
    local model="$1"
    local key="$2"
    local preset
    for preset in "$(preset_file_for_model "${model}")" "$(shipped_preset_file_for_model "${model}")"; do
        [[ -f "${preset}" ]] || continue
        local py
        py="$(command -v python 2>/dev/null || command -v python3 || true)"
        [[ -n "${py}" ]] || continue
        if MODEL="${model}" KEY="${key}" PRESET="${preset}" "${py}" - <<'PY'
import os, sys
from pathlib import Path
try:
    import yaml
except ImportError:
    sys.exit(1)
key = os.environ["KEY"].strip().lower().replace("-", "_")
bundle = yaml.safe_load(Path(os.environ["PRESET"]).read_text()) or {}
datasets = bundle.get("datasets") or {}
parts = key.split("/")
candidates = [key]
if len(parts) >= 2:
    candidates.append("/".join(parts[:2]))
sys.exit(0 if any(c in datasets for c in candidates) else 1)
PY
        then
            return 0
        fi
    done
    return 1
}

# If SKIP_IF_TUNED=1 and a preset exists for MODEL + DATASET_KEY, exit 0.
maybe_skip_if_tuned() {
    local model="$1"
    local dataset_key="$2"
    if [[ "${SKIP_IF_TUNED:-0}" != "1" ]]; then
        return 0
    fi
    if preset_has_dataset_key "${model}" "${dataset_key}"; then
        echo "SKIP_IF_TUNED=1: preset already exists for ${model} / ${dataset_key}"
        echo "  Check: ./scripts/hyperparameter/check_tuning_status.sh ${model} ${dataset_key}"
        echo "  Re-run without SKIP_IF_TUNED to search again."
        exit 0
    fi
}

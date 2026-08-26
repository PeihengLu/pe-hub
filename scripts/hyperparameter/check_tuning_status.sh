#!/usr/bin/env bash
# Check whether Optuna HPO has already written a dataset preset for a model.
#
# Checks local overlay first (training_presets_local), then shipped defaults.
#
# Usage:
#   ./scripts/hyperparameter/check_tuning_status.sh pridict2
#   ./scripts/hyperparameter/check_tuning_status.sh pridict2 minsepie/library_insert_set12
#   ./scripts/hyperparameter/check_tuning_status.sh deepprime deepprime/deepprime_clinvar/hek293t/pe2
#
# Exit codes:
#   0  preset exists (and key exists if provided)
#   1  missing / not tuned yet
#   2  usage error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

MODEL="${1:-}"
DATASET_KEY="${2:-}"

if [[ -z "${MODEL}" ]]; then
    echo "Usage: $0 <model> [dataset_key]" >&2
    echo "  dataset_key example: minsepie/library_insert_set12/hek293t/pe2" >&2
    exit 2
fi

MODEL_LC="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
LOCAL_FILE="$(preset_file_for_model "${MODEL_LC}")"
SHIPPED_FILE="$(shipped_preset_file_for_model "${MODEL_LC}")"
STUDIES_ROOT="${TUNING_STUDIES_ROOT:-${REPO_ROOT}/services/pe-ensemble/tuning_studies}"

echo "Model:         ${MODEL_LC}"
echo "Local preset:  ${LOCAL_FILE}"
echo "Shipped:       ${SHIPPED_FILE}"
echo "Studies dir:   ${STUDIES_ROOT}"
echo ""

_list_keys() {
    local preset_file="$1"
    local label="$2"
    [[ -f "${preset_file}" ]] || return 0
    echo "${label} (${preset_file}):"
    if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
        PY="$(command -v python 2>/dev/null || command -v python3)"
        PRESET_FILE="${preset_file}" "${PY}" - <<'PY'
from pathlib import Path
import os
try:
    import yaml
except ImportError:
    print("  (install PyYAML to list dataset keys; file exists)")
    raise SystemExit(0)
bundle = yaml.safe_load(Path(os.environ["PRESET_FILE"]).read_text()) or {}
datasets = bundle.get("datasets") or {}
if not datasets:
    print("  (none yet — defaults only)")
else:
    for key, entry in datasets.items():
        prov = (entry or {}).get("provenance") or {}
        metric = prov.get("metric_value")
        source = prov.get("source", "?")
        searched = prov.get("searched_at", "")
        extra = f"  metric={metric}" if metric is not None else ""
        print(f"  - {key}  [{source}]{extra}  {searched}")
PY
    else
        grep -E '^\s+[a-z0-9_/]+:' "${preset_file}" || true
    fi
}

if [[ ! -f "${LOCAL_FILE}" && ! -f "${SHIPPED_FILE}" ]]; then
    echo "Status: NOT TUNED (no local or shipped preset YAML)"
    exit 1
fi

if [[ -z "${DATASET_KEY}" ]]; then
    echo "Status: preset file(s) exist. Dataset entries:"
    _list_keys "${LOCAL_FILE}" "Local"
    _list_keys "${SHIPPED_FILE}" "Shipped"
    if [[ -d "${STUDIES_ROOT}" ]]; then
        echo ""
        echo "Optuna study DBs:"
        ls -1 "${STUDIES_ROOT}"/*.db 2>/dev/null | sed 's|.*/|  |' || echo "  (none)"
    fi
    exit 0
fi

KEY_NORM="$(echo "${DATASET_KEY}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

if preset_has_dataset_key "${MODEL_LC}" "${KEY_NORM}"; then
    PY="$(command -v python 2>/dev/null || command -v python3)"
    for PRESET_FILE in "${LOCAL_FILE}" "${SHIPPED_FILE}"; do
        [[ -f "${PRESET_FILE}" ]] || continue
        if KEY_NORM="${KEY_NORM}" PRESET_FILE="${PRESET_FILE}" "${PY}" - <<'PY'
from pathlib import Path
import os, sys
try:
    import yaml
except ImportError:
    sys.exit(2)
path = Path(os.environ["PRESET_FILE"])
bundle = yaml.safe_load(path.read_text()) or {}
datasets = bundle.get("datasets") or {}
key = os.environ["KEY_NORM"]
candidates = [key]
parts = key.split("/")
if len(parts) >= 2:
    candidates.append("/".join(parts[:2]))
hit = next((c for c in candidates if c in datasets), None)
if hit is None:
    sys.exit(1)
entry = datasets[hit] or {}
prov = entry.get("provenance") or {}
layer = "local" if "training_presets_local" in str(path) else "shipped"
print(f"Status: TUNED ({layer})")
print(f"Matched key: {hit}")
print(f"Preset file: {path}")
print(f"Source:      {prov.get('source', '?')}")
print(f"Metric:      {prov.get('metric')} = {prov.get('metric_value')}")
print(f"Best trial:  {prov.get('best_trial')}")
print(f"Searched at: {prov.get('searched_at')}")
print(f"Study:       {prov.get('study_name')}")
hps = entry.get("hyperparameters") or {}
if hps:
    print("Hyperparameters:")
    for k, v in hps.items():
        print(f"  {k}: {v}")
sys.exit(0)
PY
        then
            exit 0
        fi
    done
fi

echo "Status: NOT TUNED for key ${KEY_NORM}"
exit 1

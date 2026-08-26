#!/usr/bin/env bash
# Shared helpers for PRIDICT 2.0 transfer + ensemble reproduction.
#
# Pipeline (Mathis et al. via PE Ensemble / PE-DB):
#   1. Base train on PRIDICT library1
#   2. Base train on library1 + DeepPrime ClinVar (DeepPrime folds on overlaps)
#   3. Fine-tune both bases on library-diverse HEK and K562 → 4 models
#   4. Mean-ensemble per cell line

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_DIR="$(cd "${EXP_DIR}/../../hyperparameter" && pwd)"
# shellcheck source=../../hyperparameter/_common.sh
source "${HP_DIR}/_common.sh"

STATE_DIR="${STATE_DIR:-${EXP_DIR}/state}"
mkdir -p "${STATE_DIR}"

MODEL="${MODEL:-pridict2}"
PE_SYSTEM="${PE_SYSTEM:-pe2}"
BASE_CELL_LINE="${BASE_CELL_LINE:-hek293t}"
FT_CELL_LINES="${FT_CELL_LINES:-hek k562}"

# Stable dataset-name labels (also used as state keys).
NAME_BASE_L1="${NAME_BASE_L1:-pridict2-repro-base-library1}"
NAME_BASE_L1C="${NAME_BASE_L1C:-pridict2-repro-base-l1-clinvar}"
NAME_FT_PREFIX="${NAME_FT_PREFIX:-pridict2-repro-ft}"

# Only pridict1/library1 has the full outcome trio. KL/CE is refused by peen when
# any of those columns is missing or NaN — inject MSEloss for merge / diverse FT.
force_mse_loss_json() {
    local raw="${1-}"
    if [[ -z "${raw}" ]]; then
        raw="{}"
    fi
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    PE_HUB_HP_JSON_IN="${raw}" "${py}" - <<'PY'
import json
import os

raw = (os.environ.get("PE_HUB_HP_JSON_IN") or "{}").strip() or "{}"
data = json.loads(raw)
data["loss_func"] = "MSEloss"
data["y_ref"] = ["averageedited"]
print(json.dumps(data, separators=(",", ":")))
PY
}

state_path() {
    echo "${STATE_DIR}/$1"
}

write_state() {
    local key="$1"
    local value="$2"
    printf '%s\n' "${value}" > "$(state_path "${key}")"
    echo "Wrote state ${key}=${value}"
}

read_state() {
    local key="$1"
    local path
    path="$(state_path "${key}")"
    if [[ ! -f "${path}" ]]; then
        echo "Error: missing state '${key}' at ${path}" >&2
        echo "Run the earlier pipeline stage first (or set STATE_DIR)." >&2
        exit 1
    fi
    cat "${path}"
}

# Extract the last JSON object's weights_id from peen stdout.
extract_weights_id() {
    local py
    py="$(command -v python 2>/dev/null || command -v python3)"
    "${py}" - <<'PY'
import json, sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
last = None
idx = 0
while True:
    start = text.find("{", idx)
    if start < 0:
        break
    try:
        obj, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError:
        idx = start + 1
        continue
    if isinstance(obj, dict) and obj.get("weights_id"):
        last = obj["weights_id"]
    idx = end
if not last:
    sys.stderr.write("Error: no weights_id found in peen output\n")
    sys.exit(1)
print(last)
PY
}

run_peen_capture_weights() {
    # Usage: run_peen_capture_weights <state_key> peen args...
    local state_key="$1"
    shift
    local logfile
    logfile="$(state_path "${state_key}.log")"
    echo "+ peen $*"
    echo "  (log: ${logfile})"
    if peen "$@" 2>&1 | tee "${logfile}" | extract_weights_id > "$(state_path "${state_key}.tmp")"; then
        mv "$(state_path "${state_key}.tmp")" "$(state_path "${state_key}")"
        echo "Wrote state ${state_key}=$(cat "$(state_path "${state_key}")")"
    else
        rm -f "$(state_path "${state_key}.tmp")"
        echo "Error: peen failed; see ${logfile}" >&2
        exit 1
    fi
}

maybe_skip_if_state() {
    local key="$1"
    if [[ "${SKIP_IF_DONE:-0}" != "1" ]]; then
        return 0
    fi
    if [[ -f "$(state_path "${key}")" ]]; then
        echo "SKIP_IF_DONE=1: state already exists for ${key}=$(cat "$(state_path "${key}")")"
        exit 0
    fi
}

print_repro_banner() {
    local title="$1"
    print_experiment_banner "${title}"
    echo "STATE_DIR: ${STATE_DIR}"
    echo "MODEL:     ${MODEL}"
    echo ""
}

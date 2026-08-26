#!/usr/bin/env bash
# Local preflight for peen tune/train/evaluate/ensemble before ARC submit.
#
# Builds a tiny DATA_ROOT from real standardized sheets, isolates presets /
# weights / Optuna DBs / jobs under a scratch dir, then runs the same CLI
# shapes used by pridict2-reproduction (single-sheet HPO, merge+author-fold
# HPO, train, fine-tune, evaluate, ensemble).
#
# Usage:
#   ./scripts/cluster/oxford-arc/preflight.sh
#   DEVICE=cuda:0 N_ROWS=48 ./scripts/cluster/oxford-arc/preflight.sh
#   KEEP_WORK=1 ./scripts/cluster/oxford-arc/preflight.sh   # retain scratch
#
# Env:
#   DEVICE          default auto
#   N_ROWS          rows per mini sheet (default 64)
#   N_TRIALS        Optuna trials (default 2)
#   CV_FOLDS        default 2
#   WORK_DIR        scratch root (default: /tmp/pe-hub-preflight-<pid>)
#   KEEP_WORK=1     do not delete WORK_DIR on success
#   SOURCE_DATASETS default <repo>/datasets

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ARC_DIR}/../../.." && pwd)"

if ! command -v peen >/dev/null 2>&1; then
    echo "Error: peen not on PATH. conda activate <env> && ./scripts/install-clis.sh" >&2
    exit 1
fi
if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python required" >&2
    exit 1
fi
PY="$(command -v python 2>/dev/null || command -v python3)"

DEVICE="${DEVICE:-auto}"
N_ROWS="${N_ROWS:-64}"
N_TRIALS="${N_TRIALS:-2}"
CV_FOLDS="${CV_FOLDS:-2}"
SOURCE_DATASETS="${SOURCE_DATASETS:-${REPO_ROOT}/datasets}"
WORK_DIR="${WORK_DIR:-/tmp/pe-hub-preflight-$$}"
KEEP_WORK="${KEEP_WORK:-0}"
FIXED_HP_JSON="${FIXED_HP_JSON:-{\"num_epochs\":2,\"batch_size\":16,\"lr\":0.0001}}"
# Only pridict1/library1 has the outcome trio. Merge + library-diverse use MSE.
MERGE_HP_JSON="${MERGE_HP_JSON:-{\"num_epochs\":2,\"batch_size\":16,\"lr\":0.0001,\"loss_func\":\"MSEloss\",\"y_ref\":[\"averageedited\"]}}"
FT_HP_JSON="${FT_HP_JSON:-{\"num_epochs\":2,\"batch_size\":16,\"lr\":0.0001,\"loss_func\":\"MSEloss\",\"y_ref\":[\"averageedited\"]}}"
REPO_LOCAL_PRESETS="${REPO_ROOT}/services/pe-ensemble/config/training_presets_local"

MODEL=pridict2
PE_SYSTEM=pe2
BASE_CELL=hek293t
FT_CELL=hek

cleanup() {
    local code=$?
    if [[ "${KEEP_WORK}" == "1" ]]; then
        echo ""
        echo "KEEP_WORK=1: scratch left at ${WORK_DIR}"
    else
        rm -rf "${WORK_DIR}"
    fi
    exit "${code}"
}
trap cleanup EXIT

mkdir -p "${WORK_DIR}"
echo "======================================"
echo "PE Hub peen preflight"
echo "======================================"
echo "WORK_DIR:  ${WORK_DIR}"
echo "DEVICE:    ${DEVICE}"
echo "N_ROWS:    ${N_ROWS}"
echo "N_TRIALS:  ${N_TRIALS}"
echo "CV_FOLDS:  ${CV_FOLDS}"
echo ""

echo "---- 0) Build mini DATA_ROOT ----"
"${PY}" "${ARC_DIR}/preflight_build_mini_data.py" \
    --work-dir "${WORK_DIR}" \
    --source-datasets "${SOURCE_DATASETS}" \
    --n-rows "${N_ROWS}"

# Isolate all mutable peen artifacts inside the scratch tree.
export DATA_ROOT="${WORK_DIR}/datasets"
export TRAINING_PRESETS_ROOT="${WORK_DIR}/presets_local"
export TRAINING_SHIPPED_PRESETS_ROOT="${REPO_ROOT}/services/pe-ensemble/config/training_presets"
export TUNING_STUDIES_ROOT="${WORK_DIR}/tuning_studies"
export TRAINING_JOBS_ROOT="${WORK_DIR}/jobs"
export TUNING_JOBS_ROOT="${WORK_DIR}/tune_jobs"
export WEIGHTS_ROOT="${WORK_DIR}/weights"
STATE_DIR="${WORK_DIR}/state"
mkdir -p \
    "${TRAINING_PRESETS_ROOT}" \
    "${TUNING_STUDIES_ROOT}" \
    "${TRAINING_JOBS_ROOT}" \
    "${TUNING_JOBS_ROOT}" \
    "${WEIGHTS_ROOT}" \
    "${STATE_DIR}"

# Minimal weights index (trained runs register here; no vendor trees required).
printf '%s\n' '{"schema_version":1,"entries":[]}' > "${WEIGHTS_ROOT}/registry.json"
printf '%s\n' '{"schema_version":1,"entries":[]}' > "${WEIGHTS_ROOT}/local_registry.json"

run() {
    local title="$1"
    shift
    echo ""
    echo "---- ${title} ----"
    echo "+ $*"
    "$@"
}

extract_weights_id() {
    # Must read peen stdout from the pipe (do not use a stdin heredoc here).
    "${PY}" -c 'import re,sys
text=sys.stdin.read()
matches=re.findall(r"\"weights_id\"\s*:\s*\"([^\"]+)\"", text)
if not matches:
    sys.stderr.write("Error: no weights_id in peen output\n")
    sys.exit(1)
print(matches[-1])'
}

capture_weights() {
    local key="$1"
    shift
    local log="${STATE_DIR}/${key}.log"
    echo "+ peen $*"
    if peen "$@" 2>&1 | tee "${log}" | extract_weights_id > "${STATE_DIR}/${key}.tmp"; then
        mv "${STATE_DIR}/${key}.tmp" "${STATE_DIR}/${key}"
        echo "  weights_id=$(cat "${STATE_DIR}/${key}")"
    else
        rm -f "${STATE_DIR}/${key}.tmp"
        echo "Error: peen failed for ${key}; see ${log}" >&2
        exit 1
    fi
}

# Assert train/tune log JSON contains pretrained_load.skipped_mismatched including decoder
# (trio base → edit-only FT) or an empty skip list (same-shape transfer).
assert_pretrained_load() {
    local log="$1"
    local expect_skip_decoder="$2" # 1 = decoder must be skipped; 0 = decoder must load
    "${PY}" - "${log}" "${expect_skip_decoder}" <<'PY'
import json, sys
from pathlib import Path

log_path = Path(sys.argv[1])
expect_skip = sys.argv[2] == "1"
text = log_path.read_text(encoding="utf-8")
decoder = json.JSONDecoder()
payload = None
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
    if isinstance(obj, dict) and (
        "pretrained_load" in obj
        or (isinstance(obj.get("result"), dict) and "pretrained_load" in obj["result"])
    ):
        payload = obj.get("pretrained_load") or obj["result"]["pretrained_load"]
    idx = end

if payload is None:
    sys.stderr.write(f"Error: no pretrained_load in {log_path}\n")
    sys.exit(1)

loaded = set(payload.get("loaded") or [])
skipped = set(payload.get("skipped_mismatched") or [])
print(f"  pretrained_load loaded={sorted(loaded)}")
print(f"  pretrained_load skipped_mismatched={sorted(skipped)}")

if expect_skip:
    if "decoder" not in skipped:
        sys.stderr.write(
            "Error: expected decoder in skipped_mismatched for trio→edit-only FT\n"
        )
        sys.exit(1)
    if "init_encoder" not in loaded:
        sys.stderr.write("Error: expected backbone init_encoder to load\n")
        sys.exit(1)
else:
    if "decoder" in skipped:
        sys.stderr.write("Error: same-shape FT unexpectedly skipped decoder\n")
        sys.exit(1)
    if "decoder" not in loaded:
        sys.stderr.write("Error: expected decoder to load for same-shape FT\n")
        sys.exit(1)
print("  pretrained_load check OK")
PY
}

run "1) peen devices / models" peen devices
run "1b) peen models" peen models

# --- Single-sheet tune (same shape as 01_tune_base_library1) ---
capture_weights base_library1_tune \
    tune \
    --model "${MODEL}" \
    --dataset-name "preflight-base-library1" \
    --study pridict1 --dataset library1 \
    --cell-line "${BASE_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy cv \
    --cv-folds "${CV_FOLDS}" \
    --test-pct 0.2 \
    --split-random-state 42 \
    --n-trials "${N_TRIALS}" \
    --study-name "preflight__base_library1" \
    --device "${DEVICE}" \
    --fixed-hyperparameters-json "${FIXED_HP_JSON}" \
    --register-best-weights \
    --notes "preflight: single-sheet HPO"

# --- Single-sheet train (03) ---
capture_weights base_library1 \
    train \
    --model "${MODEL}" \
    --dataset-name "preflight-train-library1" \
    --study pridict1 --dataset library1 \
    --cell-line "${BASE_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy holdout_3 \
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15 \
    --split-random-state 42 \
    --device "${DEVICE}" \
    --hyperparameters-json "${FIXED_HP_JSON}" \
    --notes "preflight: train library1"

# --- Merge train (04 shape) — exercises --merge + seq_id remapping ---
# Skip merge *tune* on the mini sample (tiny folds). Full-data ARC: stage 02.
# MSEloss required: ClinVar has edit efficiency only (no outcome trio).
capture_weights base_l1_clinvar \
    train \
    --model "${MODEL}" \
    --dataset-name "preflight-train-l1-clinvar" \
    --study pridict1 --dataset library1 \
    --study deepprime --dataset deepprime-clinvar \
    --cell-line "${BASE_CELL}" --pe-system "${PE_SYSTEM}" \
    --merge \
    --split-strategy holdout_3 \
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15 \
    --split-random-state 42 \
    --device "${DEVICE}" \
    --hyperparameters-json "${MERGE_HP_JSON}" \
    --notes "preflight: train merge L1+ClinVar (MSEloss)"

BASE_L1="$(cat "${STATE_DIR}/base_library1")"
BASE_L1C="$(cat "${STATE_DIR}/base_l1_clinvar")"

# --- Fine-tune with --pretrained-weights (06 shape) ---
# Trio library1 base (KLDloss, num_outcomes=3) → library-diverse MSE (1 outcome):
# backbone loads; decoder is skipped (shape mismatch) and re-initialized.
capture_weights "ft_base_library1_${FT_CELL}" \
    train \
    --model "${MODEL}" \
    --dataset-name "preflight-ft-l1-${FT_CELL}" \
    --study pridict2 --dataset library-diverse \
    --cell-line "${FT_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy holdout_3 \
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15 \
    --split-random-state 42 \
    --device "${DEVICE}" \
    --pretrained-weights "${BASE_L1}" \
    --hyperparameters-json "${FT_HP_JSON}" \
    --notes "preflight: FT library1 trio→diverse (skip mismatched decoder)"

assert_pretrained_load "${STATE_DIR}/ft_base_library1_${FT_CELL}.log" 1

# Merge base was already MSEloss / 1 outcome → same-shape transfer (decoder loads).
capture_weights "ft_base_l1_clinvar_${FT_CELL}" \
    train \
    --model "${MODEL}" \
    --dataset-name "preflight-ft-l1c-${FT_CELL}" \
    --study pridict2 --dataset library-diverse \
    --cell-line "${FT_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy holdout_3 \
    --train-pct 0.7 --val-pct 0.15 --test-pct 0.15 \
    --split-random-state 43 \
    --device "${DEVICE}" \
    --pretrained-weights "${BASE_L1C}" \
    --hyperparameters-json "${FT_HP_JSON}" \
    --notes "preflight: FT merge-base→diverse (same-shape decoder loads)"

assert_pretrained_load "${STATE_DIR}/ft_base_l1_clinvar_${FT_CELL}.log" 0

FT1="$(cat "${STATE_DIR}/ft_base_library1_${FT_CELL}")"
FT2="$(cat "${STATE_DIR}/ft_base_l1_clinvar_${FT_CELL}")"

# --- Evaluate ---
run "evaluate FT1" \
    peen evaluate \
    --model "${MODEL}" \
    --weights "${FT1}" \
    --study pridict2 --dataset library-diverse \
    --cell-line "${FT_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy holdout_2 \
    --train-pct 0.8 --test-pct 0.2 \
    --split-random-state 42 \
    --device "${DEVICE}" \
    --sync

# --- Ensemble (07 shape) ---
run "ensemble mean" \
    peen ensemble \
    --ensemble-name "preflight-ensemble-${FT_CELL}" \
    --combine mean \
    --member "${MODEL}:${FT1}" \
    --member "${MODEL}:${FT2}" \
    --study pridict2 --dataset library-diverse \
    --cell-line "${FT_CELL}" --pe-system "${PE_SYSTEM}" \
    --split-strategy holdout_2 \
    --train-pct 0.8 --test-pct 0.2 \
    --no-use-original-fold \
    --split-random-state 42 \
    --device "${DEVICE}" \
    --sync

# --- Artifact checks ---
echo ""
echo "---- checks ----"
run "scratch local presets written (isolated TRAINING_PRESETS_ROOT)" \
    "${REPO_ROOT}/scripts/hyperparameter/check_tuning_status.sh" "${MODEL}"
test -f "${TRAINING_PRESETS_ROOT}/${MODEL}.yaml"
test -s "${STATE_DIR}/base_library1"
test -s "${STATE_DIR}/base_l1_clinvar"
run "list weights" peen weights --model "${MODEL}"

echo ""
echo "======================================"
echo "PREFLIGHT PASSED"
echo "======================================"
echo "Covered:"
echo "  - peen devices/models"
echo "  - tune single-sheet (cv) + register-best"
echo "  - train single-sheet (holdout_3)"
echo "  - train merge L1+ClinVar (seq_id remapping, MSEloss)"
echo "  - --pretrained-weights FT trio→edit-only (decoder skipped)"
echo "  - --pretrained-weights FT same-shape (decoder loaded)"
echo "  - evaluate + mean ensemble"
echo "  - scratch local presets overlay write"
echo ""
echo "Local presets note:"
echo "  Preflight wrote to: ${TRAINING_PRESETS_ROOT}/"
echo "  Repo path (gitignored, used by normal peen): ${REPO_LOCAL_PRESETS}/"
echo "  Preflight does NOT touch the repo local presets dir."
echo ""
echo "Deferred to full-data ARC runs:"
echo "  - merge HPO + --use-original-fold (stage 02; MSEloss)"
if [[ "${KEEP_WORK}" == "1" ]]; then
    echo "Scratch: ${WORK_DIR}"
fi

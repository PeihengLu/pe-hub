#!/usr/bin/env bash
# Run the full PRIDICT 2.0 transfer + ensemble reproduction pipeline.
#
# Stages:
#   01 tune base library1
#   02 tune base library1 + DeepPrime ClinVar
#   03 train base library1
#   04 train base L1+ClinVar
#   05 tune fine-tune (library-diverse HEK / K562)
#   06 fine-tune 4 transfer models
#   07 mean-ensemble per cell line
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/run_all.sh
#   SMOKE=1 DEVICE=mps SKIP_IF_TUNED=1 SKIP_IF_DONE=1 ./.../run_all.sh
#   ONLY=06,07 ./.../run_all.sh          # resume from fine-tune + ensemble
#   SKIP=01,02,05 ./.../run_all.sh       # skip HPO (use existing presets)
#
# Env (also see _common.sh / scripts/hyperparameter/_common.sh):
#   DEVICE, N_TRIALS, SMOKE, SKIP_IF_TUNED, SKIP_IF_DONE, STATE_DIR
#   FT_CELL_LINES   default "hek k562"
#   ONLY / SKIP     comma-separated stage numbers (01..07)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "PRIDICT 2.0 reproduction — full pipeline"

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

should_run() {
    local stage="$1"
    if [[ -n "${ONLY}" ]]; then
        [[ ",${ONLY}," == *",${stage},"* ]] || return 1
    fi
    if [[ -n "${SKIP}" ]]; then
        [[ ",${SKIP}," == *",${stage},"* ]] && return 1
    fi
    return 0
}

run_stage() {
    local stage="$1"
    local script="$2"
    if ! should_run "${stage}"; then
        echo "---- skip stage ${stage} (${script}) ----"
        return 0
    fi
    echo ""
    echo "######## STAGE ${stage}: ${script} ########"
    bash "${SCRIPT_DIR}/${script}"
}

# Default: skip retuning / retraining when artifacts already exist.
export SKIP_IF_TUNED="${SKIP_IF_TUNED:-1}"
export SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

run_stage 01 "01_tune_base_library1.sh"
run_stage 02 "02_tune_base_l1_clinvar.sh"
run_stage 03 "03_train_base_library1.sh"
run_stage 04 "04_train_base_l1_clinvar.sh"
run_stage 05 "05_tune_finetune_library_diverse.sh"
run_stage 06 "06_finetune_transfer.sh"
run_stage 07 "07_ensemble_by_cell_line.sh"

echo ""
echo "======================================"
echo "PRIDICT 2.0 reproduction finished"
echo "State directory: ${STATE_DIR}"
echo "======================================"
if [[ -d "${STATE_DIR}" ]]; then
    ls -1 "${STATE_DIR}" | sed 's/^/  /'
fi

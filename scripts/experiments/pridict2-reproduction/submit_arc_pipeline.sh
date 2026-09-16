#!/usr/bin/env bash
# Submit PRIDICT2 reproduction stages to ARC with SLURM --dependency=afterok.
#
# DAG (01∥02, then trains, then FT tune → transfer → ensemble):
#
#   01 tune L1 ──────────────► 03 train L1 ──┐
#                                             ├─► 05 tune FT ──► 06 FT ×4 ──► 07 ensemble
#   02 tune L1+ClinVar ──────► 04 train L1C ─┘
#                              (06 also waits on 04)
#
# Usage (on htc-login, from repo root):
#   ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
#
#   # Skip HPO if presets already exist; only chain train → FT
#   SKIP=01,02 ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
#
#   ONLY=01,03 ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
#
# Per-stage walltime (override via env):
#   ARC_TIME_01 / ARC_PARTITION_01 … ARC_TIME_07 / ARC_PARTITION_07
# Defaults: 01 → medium 24h; 02 → medium 48h (merged L1+ClinVar holdout_3); 03–07 → short 12h.
#
# Requires: scripts/cluster/oxford-arc/env.sh (same as submit.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARC_SUBMIT="${REPO_ROOT}/scripts/cluster/oxford-arc/submit.sh"

if [[ ! -x "${ARC_SUBMIT}" ]]; then
    echo "Error: ${ARC_SUBMIT} not found or not executable" >&2
    exit 1
fi

ONLY="${ONLY:-}"
SKIP="${SKIP:-}"

export SKIP_IF_TUNED="${SKIP_IF_TUNED:-1}"
export SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

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

afterok_deps() {
    local ids=() id
    for id in "$@"; do
        [[ -n "${id}" ]] || continue
        ids+=("${id}")
    done
    if ((${#ids[@]} == 0)); then
        echo ""
        return 0
    fi
    local joined
    joined="$(IFS=:; echo "${ids[*]}")"
    echo "afterok:${joined}"
}

# Args: stage_num script [dependency] [partition] [time]
# Prints job id on stdout (empty if skipped).
submit_stage() {
    local stage="$1"
    local script="$2"
    local dep="${3:-}"
    local partition="${4:-}"
    local walltime="${5:-}"

    if ! should_run "${stage}"; then
        echo "---- skip stage ${stage} (${script}) ----" >&2
        echo ""
        return 0
    fi

    local -a env_args=()
    [[ -n "${partition}" ]] && env_args+=("ARC_PARTITION=${partition}")
    [[ -n "${walltime}" ]] && env_args+=("ARC_TIME=${walltime}")
    [[ -n "${dep}" ]] && env_args+=("ARC_DEPENDENCY=${dep}")

    echo "" >&2
    echo "######## STAGE ${stage}: ${script} ########" >&2
    if [[ -n "${dep}" ]]; then
        echo "dependency: ${dep}" >&2
    fi

    local out job_id
    if ((${#env_args[@]} > 0)); then
        out="$(env "${env_args[@]}" "${ARC_SUBMIT}" "${script}" 2>&1)" || {
            echo "${out}" >&2
            echo "Error: submit failed for ${script}" >&2
            exit 1
        }
    else
        out="$("${ARC_SUBMIT}" "${script}" 2>&1)" || {
            echo "${out}" >&2
            echo "Error: submit failed for ${script}" >&2
            exit 1
        }
    fi
    echo "${out}" >&2
    job_id="$(echo "${out}" | tail -n 1 | tr -d '[:space:]')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Error: could not parse job id from submit output (got '${job_id}')" >&2
        exit 1
    fi
    echo "${job_id}"
}

: "${ARC_PARTITION_01:=${ARC_PARTITION_MEDIUM:-medium}}"
: "${ARC_TIME_01:=${ARC_TIME_MEDIUM:-24:00:00}}"
: "${ARC_PARTITION_02:=${ARC_PARTITION_MEDIUM:-medium}}"
: "${ARC_TIME_02:=48:00:00}"
: "${ARC_PARTITION_03:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_03:=${ARC_TIME_SHORT:-12:00:00}}"
: "${ARC_PARTITION_04:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_04:=${ARC_TIME_SHORT:-12:00:00}}"
: "${ARC_PARTITION_05:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_05:=${ARC_TIME_SHORT:-12:00:00}}"
: "${ARC_PARTITION_06:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_06:=${ARC_TIME_SHORT:-12:00:00}}"
: "${ARC_PARTITION_07:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_07:=${ARC_TIME_SHORT:-06:00:00}}"

echo "======================================"
echo "PRIDICT2 reproduction — ARC dependency pipeline"
echo "======================================"
echo "SKIP_IF_TUNED=${SKIP_IF_TUNED}  SKIP_IF_DONE=${SKIP_IF_DONE}"
echo "ONLY=${ONLY:-*}  SKIP=${SKIP:-(none)}"
echo ""

JOB01="$(submit_stage 01 01_tune_base_library1.sh "" "${ARC_PARTITION_01}" "${ARC_TIME_01}")"
JOB02="$(submit_stage 02 02_tune_base_l1_clinvar.sh "" "${ARC_PARTITION_02}" "${ARC_TIME_02}")"

JOB03="$(submit_stage 03 03_train_base_library1.sh \
    "$(afterok_deps "${JOB01}")" "${ARC_PARTITION_03}" "${ARC_TIME_03}")"
JOB04="$(submit_stage 04 04_train_base_l1_clinvar.sh \
    "$(afterok_deps "${JOB02}")" "${ARC_PARTITION_04}" "${ARC_TIME_04}")"

JOB05="$(submit_stage 05 05_tune_finetune_library_diverse.sh \
    "$(afterok_deps "${JOB03}")" "${ARC_PARTITION_05}" "${ARC_TIME_05}")"

JOB06="$(submit_stage 06 06_finetune_transfer.sh \
    "$(afterok_deps "${JOB03}" "${JOB04}" "${JOB05}")" "${ARC_PARTITION_06}" "${ARC_TIME_06}")"

JOB07="$(submit_stage 07 07_ensemble_by_cell_line.sh \
    "$(afterok_deps "${JOB06}")" "${ARC_PARTITION_07}" "${ARC_TIME_07}")"

echo ""
echo "======================================"
echo "Submitted (empty = skipped)"
echo "======================================"
echo "  01 tune L1:           ${JOB01:-—}"
echo "  02 tune L1+ClinVar:   ${JOB02:-—}"
echo "  03 train L1:          ${JOB03:-—}  (afterok:01)"
echo "  04 train L1+ClinVar:  ${JOB04:-—}  (afterok:02)"
echo "  05 tune FT:           ${JOB05:-—}  (afterok:03)"
echo "  06 fine-tune ×4:      ${JOB06:-—}  (afterok:03,04,05)"
echo "  07 ensemble:          ${JOB07:-—}  (afterok:06)"
echo ""
echo "Monitor: squeue -u \$USER"
_cancel_ids=("${JOB01}" "${JOB02}" "${JOB03}" "${JOB04}" "${JOB05}" "${JOB06}" "${JOB07}")
_cancel_list=""
for _id in "${_cancel_ids[@]}"; do
    [[ -n "${_id}" ]] || continue
    _cancel_list+=" ${_id}"
done
if [[ -n "${_cancel_list}" ]]; then
    echo "Cancel chain: scancel${_cancel_list}"
fi

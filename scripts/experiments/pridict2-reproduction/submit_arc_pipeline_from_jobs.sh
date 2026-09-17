#!/usr/bin/env bash
# Resume PRIDICT2 reproduction from an already-queued ARC chain, submitting
# fold-matched 05/06 (one short/3h job per cell × fold) then 07.
#
# Snapshot defaults (htc, 2026-09-17 — override via env):
#
#   JOB02=8823958  medium  pe-02-tu   # 02 tune L1+ClinVar (keep)
#   JOB04=8823960  short   pe-04-tr   # 04 train L1C afterok:02 (keep)
#   CANCEL_JOBS=8823961,8823962,8823963
#     8823961  short pe-05-tu  # obsolete CV+final 05
#     8823962  short pe-06-fi  # obsolete combined 06
#     8823963  short pe-07-en  # obsolete 07
#
# Assumes stage 03 (base_library1) already finished. New DAG:
#
#   JOB02 ──afterok──► JOB04 ──► 06_{cell}_f{fold} (10) ──┐
#   (03 done) ─────────────────► 05_{cell}_f{fold} (10) ──┼─► 07
#
# Usage (on htc-login, from repo root):
#   ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline_from_jobs.sh
#
#   JOB04=8823960 CANCEL_JOBS=8823961,8823962,8823963 \
#     ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline_from_jobs.sh
#
#   DRY_RUN=1 …   # print plan + sbatch --test-only, no scancel / submit
#   SCANCEL=0 …   # do not cancel obsolete jobs
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

# --- Anchors from the live queue (override via env) ---
: "${JOB02:=8823958}"
: "${JOB04:=8823960}"
# Optional: if stage 03 is still running, set JOB03 so 05 waits on it.
: "${JOB03:=}"
: "${CANCEL_JOBS:=8823961,8823962,8823963}"

FT_CELL_LINES="${FT_CELL_LINES:-hek k562}"
FOLDS="${FOLDS:-0 1 2 3 4}"

: "${ARC_PARTITION_05:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_05:=03:00:00}"
: "${ARC_PARTITION_06:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_06:=03:00:00}"
: "${ARC_PARTITION_07:=${ARC_PARTITION_SHORT:-short}}"
: "${ARC_TIME_07:=${ARC_TIME_SHORT:-06:00:00}}"

export SKIP_IF_TUNED="${SKIP_IF_TUNED:-1}"
export SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SCANCEL="${SCANCEL:-1}"

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

# Args: label script dependency partition time [CELL_LINE] [FOLD]
submit_stage() {
    local label="$1"
    local script="$2"
    local dep="${3:-}"
    local partition="${4:-}"
    local walltime="${5:-}"
    local cell="${6:-}"
    local fold="${7:-}"

    local -a env_args=()
    [[ -n "${partition}" ]] && env_args+=("ARC_PARTITION=${partition}")
    [[ -n "${walltime}" ]] && env_args+=("ARC_TIME=${walltime}")
    [[ -n "${dep}" ]] && env_args+=("ARC_DEPENDENCY=${dep}")
    [[ -n "${cell}" ]] && env_args+=("CELL_LINE=${cell}")
    [[ -n "${fold}" ]] && env_args+=("FOLD=${fold}")

    echo "" >&2
    echo "######## ${label}: ${script}${cell:+ (CELL_LINE=${cell})}${fold:+ (FOLD=${fold})} ########" >&2
    if [[ -n "${dep}" ]]; then
        echo "dependency: ${dep}" >&2
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        env_args+=("DRY_RUN=1")
        env "${env_args[@]}" "${ARC_SUBMIT}" "${script}" >&2 || true
        echo ""
        return 0
    fi

    local out job_id
    out="$(env "${env_args[@]}" "${ARC_SUBMIT}" "${script}" 2>&1)" || {
        echo "${out}" >&2
        echo "Error: submit failed for ${script} (${label})" >&2
        exit 1
    }
    echo "${out}" >&2
    job_id="$(echo "${out}" | tail -n 1 | tr -d '[:space:]')"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Error: could not parse job id from submit output (got '${job_id}')" >&2
        exit 1
    fi
    echo "${job_id}"
}

# shellcheck disable=SC2206
_cells=(${FT_CELL_LINES})
# shellcheck disable=SC2206
_folds=(${FOLDS})

echo "======================================"
echo "PRIDICT2 reproduction — resume fold-matched FT"
echo "======================================"
echo "Anchors:  JOB02=${JOB02}  JOB03=${JOB03:-(done)}  JOB04=${JOB04}"
echo "Cancel:   CANCEL_JOBS=${CANCEL_JOBS:-"(none)"}  SCANCEL=${SCANCEL}"
echo "SKIP_IF_DONE=${SKIP_IF_DONE}  DRY_RUN=${DRY_RUN}"
echo "cells=${FT_CELL_LINES}  folds=${FOLDS}"
echo ""
echo "Plan:"
echo "  keep  02=${JOB02}  04=${JOB04}"
echo "  cancel obsolete CV+final 05–07"
echo "  add   05: ${#_cells[@]}×${#_folds[@]} Model A fold jobs (afterok:JOB03 if set)"
echo "        06: ${#_cells[@]}×${#_folds[@]} Model B fold jobs (afterok:JOB04)"
echo "        07: ensemble (afterok: all 05+06)"
echo ""

if [[ "${SCANCEL}" == "1" && -n "${CANCEL_JOBS}" ]]; then
    # shellcheck disable=SC2206
    _cancel_arr=(${CANCEL_JOBS//,/ })
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "+ scancel ${_cancel_arr[*]}  (DRY_RUN: not executed)"
    else
        echo "+ scancel ${_cancel_arr[*]}"
        scancel "${_cancel_arr[@]}" || {
            echo "Warning: scancel failed (jobs may already be gone); continuing" >&2
        }
    fi
fi

DEP_05="$(afterok_deps "${JOB03}")"
DEP_06="$(afterok_deps "${JOB04}")"

JOB05_IDS=()
JOB06_IDS=()
for cell in "${_cells[@]}"; do
    for fold in "${_folds[@]}"; do
        j05="$(submit_stage "05_${cell}_f${fold}" 05_tune_finetune_library_diverse.sh \
            "${DEP_05}" "${ARC_PARTITION_05}" "${ARC_TIME_05}" \
            "${cell}" "${fold}")"
        [[ -n "${j05}" ]] && JOB05_IDS+=("${j05}")

        j06="$(submit_stage "06_${cell}_f${fold}" 06_finetune_transfer.sh \
            "${DEP_06}" "${ARC_PARTITION_06}" "${ARC_TIME_06}" \
            "${cell}" "${fold}")"
        [[ -n "${j06}" ]] && JOB06_IDS+=("${j06}")
    done
done

DEP_07="$(afterok_deps "${JOB05_IDS[@]}" "${JOB06_IDS[@]}")"
if [[ -z "${DEP_07}" && "${DRY_RUN}" == "1" ]]; then
    DEP_07="afterok:<05_*>:<06_*>"
fi
JOB07="$(submit_stage "07 ensemble" 07_ensemble_by_cell_line.sh \
    "${DEP_07}" "${ARC_PARTITION_07}" "${ARC_TIME_07}")"

echo ""
echo "======================================"
echo "Submitted (empty = dry-run / skipped)"
echo "======================================"
echo "  (keep) 02 tune L1+ClinVar: ${JOB02}"
echo "  (keep) 04 train L1+ClinVar: ${JOB04}  (afterok:02)"
echo "  05 Model A fold FTs:       ${#JOB05_IDS[@]} jobs"
for _id in "${JOB05_IDS[@]+"${JOB05_IDS[@]}"}"; do
    echo "      ${_id}"
done
echo "  06 Model B fold FTs:       ${#JOB06_IDS[@]} jobs"
for _id in "${JOB06_IDS[@]+"${JOB06_IDS[@]}"}"; do
    echo "      ${_id}"
done
echo "  07 ensemble:               ${JOB07:-—}"
echo ""
echo "Monitor: squeue -u \$USER"
_cancel_ids=("${JOB05_IDS[@]+"${JOB05_IDS[@]}"}" "${JOB06_IDS[@]+"${JOB06_IDS[@]}"}" "${JOB07}")
_cancel_list=""
for _id in "${_cancel_ids[@]}"; do
    [[ -n "${_id}" ]] || continue
    _cancel_list+=" ${_id}"
done
if [[ -n "${_cancel_list}" ]]; then
    echo "Cancel new chain: scancel${_cancel_list}"
fi
echo ""
echo "Note: confirm state/base_library1 exists before 05 jobs start (stage 03)."

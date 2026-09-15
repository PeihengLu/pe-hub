#!/usr/bin/env bash
# Stage 05 — Optuna HPO for fine-tune stage on PRIDICT2 library-diverse.
# Runs once per cell line in FT_CELL_LINES (default: hek k562).
#
# Protocol: author 5-fold CV (testset_fold 0–4), no outer random holdout.
# After HPO, registers best-trial weights and evaluates on LD_TEST_FOLD
# (default 4 — same fold peen holds out for the final CV export).
#
# Transfer HPO: loads a base checkpoint (--pretrained-weights) so trials match
# stage 06 fine-tunes. Default base is state/base_library1 (override with
# PRETRAINED_WEIGHTS=<id>). Requires stage 03 (or PRETRAINED_WEIGHTS) first.
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/05_tune_finetune_library_diverse.sh
#   CELL_LINE=hek SMOKE=1 ./.../05_tune_finetune_library_diverse.sh   # single line
#   PRETRAINED_WEIGHTS=pridict2__... ./.../05_tune_finetune_library_diverse.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
require_peen

print_repro_banner "05 Tune fine-tune: library-diverse (author CV5, no outer test)"

PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
if [[ -z "${PRETRAINED_WEIGHTS}" ]]; then
    PRETRAINED_WEIGHTS="$(read_state base_library1)"
fi
echo "pretrained_weights: ${PRETRAINED_WEIGHTS}"
echo "LD_TEST_FOLD (post-tune evaluate): ${LD_TEST_FOLD}"
echo ""

# library-diverse is edit-efficiency only — keep MSEloss explicit for Optuna.
FIXED_HP_JSON="${FIXED_HP_JSON:-}"
if [[ "${SMOKE}" == "1" && -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON="$(smoke_fixed_hp_json)"
elif [[ -z "${FIXED_HP_JSON}" ]]; then
    FIXED_HP_JSON='{}'
fi
FIXED_HP_JSON="$(force_mse_loss_json "${FIXED_HP_JSON}")"
# load_pretrained is set by --pretrained-weights on peen tune; keep JSON clean.
FIXED_HP_JSON="$(with_load_pretrained_json "${FIXED_HP_JSON}" true)"
export FIXED_HP_JSON

if [[ -n "${CELL_LINE:-}" ]]; then
    lines=("${CELL_LINE}")
else
    # shellcheck disable=SC2206
    lines=(${FT_CELL_LINES})
fi

export NO_OUTER_TEST=1
export REGISTER_BEST_WEIGHTS="${REGISTER_BEST_WEIGHTS:-1}"

for cell in "${lines[@]}"; do
    echo "---- library-diverse / ${cell} (FT from ${PRETRAINED_WEIGHTS}) ----"
    state_key="ft_hpo_${cell}"
    logfile="$(state_path "${state_key}.log")"

    if [[ "${SKIP_IF_TUNED:-0}" == "1" && -f "$(state_path "${state_key}")" ]]; then
        echo "SKIP_IF_TUNED=1: ${state_key}=$(cat "$(state_path "${state_key}")")"
        echo "---- evaluate ${cell} on author fold ${LD_TEST_FOLD} ----"
        evaluate_library_diverse_test_fold \
            "$(cat "$(state_path "${state_key}")")" \
            "${cell}" \
            "${LD_TEST_FOLD}" \
            "ft_hpo"
        echo ""
        continue
    fi

    set +e
    STUDY_NAME="pridict2__repro_ft_library_diverse_${cell}" \
        PYTHONUNBUFFERED=1 \
        "${HP_DIR}/tune_hpo_cv5.sh" \
        --model "${MODEL}" \
        --dataset-name "${NAME_FT_PREFIX}-library-diverse-${cell}" \
        --study pridict2 --dataset library-diverse \
        --cell-line "${cell}" --pe-system "${PE_SYSTEM}" \
        --use-original-fold \
        --pretrained-weights "${PRETRAINED_WEIGHTS}" \
        >"${logfile}" 2>&1
    tune_rc=$?
    set -e

    if [[ "${tune_rc}" -ne 0 ]]; then
        echo "Error: tune failed for ${cell}; see ${logfile}" >&2
        tail -n 40 "${logfile}" >&2 || true
        exit 1
    fi

    if ! extract_weights_id "${logfile}" > "$(state_path "${state_key}.tmp")"; then
        if grep -qE 'SKIP_IF_TUNED|already (tuned|exists)' "${logfile}"; then
            echo "Tune skipped (preset exists); no new weights_id to evaluate."
            echo ""
            continue
        fi
        echo "Error: no weights_id from tune for ${cell}; see ${logfile}" >&2
        tail -n 40 "${logfile}" >&2 || true
        exit 1
    fi
    mv "$(state_path "${state_key}.tmp")" "$(state_path "${state_key}")"
    weights_id="$(cat "$(state_path "${state_key}")")"
    echo "Wrote state ${state_key}=${weights_id}"

    echo "---- evaluate ${cell} on author fold ${LD_TEST_FOLD} ----"
    evaluate_library_diverse_test_fold "${weights_id}" "${cell}" "${LD_TEST_FOLD}" "ft_hpo"
    echo ""
done

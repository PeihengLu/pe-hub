#!/usr/bin/env bash
# Evaluate base vendor weights on pooled PE-DB benchmarks via peen evaluate --sync,
# then PRIDICT2 mean ensembles via peen ensemble --sync.
#
# Weights:
#   - deepprime / DeepPrime_base
#   - oped / pegRNA_Model_Merged_saved.order3_decoder_weights
#   - optiprime / base
#   - pridict2 / mean ensemble only (pridict1_1 + pridict1_2, matching CV fold + head)
#     Single Model A / Model B checkpoints are not scored.
#
# Benchmarks (dataset lists; multi-cell-line datasets are expanded per cell):
#   minsepie-insert-pooled, deeppe-pooled, deepprime-clinvar,
#   pridict1-library1, pridict2-library-diverse, optiprime-lib-mmr, optiprime-lib-cv
#
# Splits:
#   - DeepPrime ClinVar / DeepPE: author original_fold=-1 test
#   - PRIDICT2 × library-diverse: run_x tests on author testset_fold==x
#   - PRIDICT library1 has no author split: vendor training used every locus
#   - OptiPrime has_original_test_split=false: in-domain ClinVar / library-diverse
#     / lib-* / library1 abort (Hsu pooled protospacer CV, not Yu/Mathis holdouts)
#   - everything else: random group holdout
#
# Note: peen ensemble unions member train_target_loci from weight provenance
# and applies the same leak exclude/abort policy as evaluate. Cells that fully
# overlap member training data (e.g. pridict1-library1) abort with data_leak.
#
# Usage:
#   conda activate pedb
#   DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
#
# Env:
#   DEVICE            compute device (default: auto)
#   RUN_ID            output run id (default: UTC timestamp). Reuse an existing id
#                     to overwrite matching cells and refresh summary.csv.
#   OUT_ROOT          results root (default: scripts/experiments/base-model-eval/results)
#   MODELS            comma/space list to restrict weights (deepprime,oped,optiprime,pridict2).
#                     pridict2 selects ensembles only (no single A/B evaluate jobs).
#                     Leaving MODELS unset runs the three vendor models plus ensembles.
#   BENCHMARKS        comma/space list of benchmark names. Extra (not in the
#                     default heatmap matrix): deeppe-ht-test (Liu Fig. 2a HT-only).
#   CELL_LINES        comma/space list of expanded cell-line names to keep
#                     (e.g. hek293t,k562). Empty keeps every expanded cell.
#   ALLOW_DATA_LEAK=1 pass --allow-data-leak (keep train-overlapping test loci
#                     instead of excluding them / aborting no_original_test_split)
#   PRIDICT2_HEADS    comma/space list of cell-type heads (default: HEK,K562).
#                     Each PRIDICT2 CV run is scored with every listed head on
#                     every benchmark (cross-cell, not just the matching head).
#   MATCH_PRIDICT2_HEAD_TO_CELL=1  ensemble only HEK→hek293t and K562→k562
#   SKIP_EXISTING=1   skip cells already ok in results.jsonl
#                     (library-diverse fold-matched cells are distinct from
#                     earlier random-holdout rows of the same weight)
#   SMOKE=1           evaluate only first remaining weight × first remaining benchmark;
#                     ensemble stage also limited to first head × run_0 × that bench
#   DESIGN_RULESET    well-designed pegRNA filter: optiprime (Hsu PBS=13 +
#                     homology + first RTT not C) or anzalone (Anzalone 2019).
#                     Use a new RUN_ID so unfiltered cells are not skipped.
#
# Partial rerun into an existing run (DeepPrime numbers stay; others refresh):
#   DEVICE=cuda:0 MODELS=oped,pridict2,optiprime RUN_ID=20260902T151810Z \
#     ./scripts/experiments/evaluate_base_model_benchmarks.sh
#
# PRIDICT2 ensembles only (no other vendors):
#   DEVICE=cuda:0 MODELS=pridict2 SKIP_EXISTING=1 RUN_ID=20260903T133425Z \
#     ./scripts/experiments/evaluate_base_model_benchmarks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../hyperparameter/_common.sh
source "${REPO_ROOT}/scripts/hyperparameter/_common.sh"
require_peen

# Prefer module invocation: the ``peen`` console script can segfault under some
# shell/completion environments while ``python -m pe_ensemble.cli`` is reliable.
PEEN_CMD=(python -m pe_ensemble.cli)

DEVICE="${DEVICE:-auto}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/base-model-eval/results}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
RESULTS_JSONL="${OUT_DIR}/results.jsonl"
LOG_DIR="${OUT_DIR}/logs"
FILTER_MODELS="${MODELS:-}"
FILTER_BENCHMARKS="${BENCHMARKS:-}"
FILTER_CELL_LINES="${CELL_LINES:-}"
ALLOW_DATA_LEAK="${ALLOW_DATA_LEAK:-0}"
MATCH_PRIDICT2_HEAD_TO_CELL="${MATCH_PRIDICT2_HEAD_TO_CELL:-0}"
DESIGN_RULESET="${DESIGN_RULESET:-}"
WANT_PRIDICT2_ENSEMBLE=1
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

DESIGN_RULE_ARGS=()
if [[ -n "${DESIGN_RULESET}" ]]; then
  DESIGN_RULE_ARGS+=(--design-ruleset "${DESIGN_RULESET}")
fi

LEAK_ARGS=()
if [[ "${ALLOW_DATA_LEAK}" == "1" ]]; then
  LEAK_ARGS+=(--allow-data-leak)
fi

print_experiment_banner "Base model evaluation (pooled benchmarks)"
echo "RUN_ID:    ${RUN_ID}"
echo "OUT_DIR:   ${OUT_DIR}"
echo "DEVICE:    ${DEVICE}"
echo "PEEN_CMD:  ${PEEN_CMD[*]}"
echo "ALLOW_DATA_LEAK: ${ALLOW_DATA_LEAK}"
echo "DESIGN_RULESET: ${DESIGN_RULESET:-none}"
echo "MATCH_PRIDICT2_HEAD_TO_CELL: ${MATCH_PRIDICT2_HEAD_TO_CELL}"
echo ""

# model|weights|experiment_id|cv_run
WEIGHTS=(
  "deepprime|DeepPrime_base|DeepPrime_base|"
  "oped|pegRNA_Model_Merged_saved.order3_decoder_weights|oped_merged|"
  "optiprime|base|optiprime_base|"
)

# August 2023 CV experiments (each is a 5-fold run with HEK+K562 heads):
#   pridict1_1 = Model A: library1 base → library-diverse FT
#   pridict1_2 = Model B: library1+ClinVar base → library-diverse FT (HEKschwank init)
PRIDICT2_EXPERIMENTS=(
  "pridict1_1__exp_2023-08-25_20-55-53"
  "pridict1_2__exp_2023-08-28_22-22-26"
)
# Multi-head vendor runs require a cell-type suffix. Ensemble each listed
# head on every benchmark (HEK and K562 by default) so HEK-trained vs
# K562-trained decoders can be compared on the same sheet. CV mean/std is
# over run_0..4 within each head.
PRIDICT2_HEADS_KEY="$(echo "${PRIDICT2_HEADS:-HEK,K562}" | tr -s ' ,' ',' | sed 's/^,//;s/,$//')"
IFS=',' read -ra PRIDICT2_HEAD_ARR <<< "${PRIDICT2_HEADS_KEY}"
if [[ ${#PRIDICT2_HEAD_ARR[@]} -eq 0 ]]; then
  echo "Error: PRIDICT2_HEADS='${PRIDICT2_HEADS:-}' produced no heads" >&2
  exit 1
fi

# name|study|dataset1,dataset2,...
BENCHMARKS=(
  "minsepie-insert-pooled|minsepie|library-insert-set12,library-insert-18nt,library-insert-codon-variant,library-insert-codon-hek3"
  "deeppe-pooled|deeppe|deeppe-ht,deeppe-type,deeppe-position,deeppe-endo"
  "deepprime-clinvar|deepprime|deepprime-clinvar"
  "pridict1-library1|pridict1|library1"
  "pridict2-library-diverse|pridict2|library-diverse"
  "optiprime-lib-mmr|optiprime|lib-mmr"
  "optiprime-lib-cv|optiprime|lib-cv"
)

if [[ -n "${FILTER_MODELS}" ]]; then
  MODELS_KEY=",$(echo "${FILTER_MODELS}" | tr -s ' ,' ',' | sed 's/^,//;s/,$//'),"
  FILTERED_WEIGHTS=()
  for spec in "${WEIGHTS[@]}"; do
    IFS='|' read -r name _ <<< "${spec}"
    if [[ "${MODELS_KEY}" == *",${name},"* ]]; then
      FILTERED_WEIGHTS+=("${spec}")
    fi
  done
  WEIGHTS=("${FILTERED_WEIGHTS[@]}")
  WANT_PRIDICT2_ENSEMBLE=0
  if [[ "${MODELS_KEY}" == *",pridict2,"* ]]; then
    WANT_PRIDICT2_ENSEMBLE=1
  fi
  if [[ ${#WEIGHTS[@]} -eq 0 && "${WANT_PRIDICT2_ENSEMBLE}" != "1" ]]; then
    echo "Error: MODELS='${FILTER_MODELS}' matched no weights (deepprime|oped|optiprime|pridict2)" >&2
    exit 1
  fi
fi

# Not in the default heatmap matrix; selectable via BENCHMARKS=deeppe-ht-test.
EXTRA_BENCHMARKS=(
  "deeppe-ht-test|deeppe|deeppe-ht"
)

if [[ -n "${FILTER_BENCHMARKS}" ]]; then
  BENCH_KEY=",$(echo "${FILTER_BENCHMARKS}" | tr -s ' ,' ',' | sed 's/^,//;s/,$//'),"
  FILTERED_BENCHES=()
  for spec in "${BENCHMARKS[@]}"; do
    IFS='|' read -r name _ <<< "${spec}"
    if [[ "${BENCH_KEY}" == *",${name},"* ]]; then
      FILTERED_BENCHES+=("${spec}")
    fi
  done
  for spec in "${EXTRA_BENCHMARKS[@]}"; do
    IFS='|' read -r name _ <<< "${spec}"
    if [[ "${BENCH_KEY}" == *",${name},"* ]]; then
      FILTERED_BENCHES+=("${spec}")
    fi
  done
  BENCHMARKS=("${FILTERED_BENCHES[@]}")
  if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    echo "Error: BENCHMARKS='${FILTER_BENCHMARKS}' matched no benchmarks" >&2
    exit 1
  fi
fi

# Split multi-cell-line (and multi-PE) datasets so PE2/PE4 are not pooled.
mapfile -t BENCHMARKS < <(
  python "${SCRIPT_DIR}/expand_eval_cell_lines.py" \
    --datasets-dir "${REPO_ROOT}/datasets" \
    "${BENCHMARKS[@]}"
)
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  echo "Error: no benchmarks after cell-line expansion" >&2
  exit 1
fi

if [[ -n "${FILTER_CELL_LINES}" ]]; then
  CELL_KEY=",$(echo "${FILTER_CELL_LINES}" | tr '[:upper:]' '[:lower:]' | tr -s ' ,' ',' | sed 's/^,//;s/,$//'),"
  FILTERED_CELLS=()
  for spec in "${BENCHMARKS[@]}"; do
    IFS='|' read -r _name _study _datasets cell _pe <<< "${spec}"
    cell_lc="$(echo "${cell}" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "${cell}" || "${CELL_KEY}" == *",${cell_lc},"* ]]; then
      FILTERED_CELLS+=("${spec}")
    fi
  done
  BENCHMARKS=("${FILTERED_CELLS[@]}")
  if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    echo "Error: CELL_LINES='${FILTER_CELL_LINES}' matched no expanded benchmarks" >&2
    exit 1
  fi
fi

pridict2_head_matches_cell() {
  local head_u cell_n
  head_u="$(echo "${1}" | tr '[:lower:]' '[:upper:]')"
  cell_n="$(echo "${2}" | tr '[:upper:]' '[:lower:]' | tr -d '_-')"
  case "${head_u}" in
    HEK|HEK293T)
      [[ "${cell_n}" == "hek293t" || "${cell_n}" == "hek" ]]
      ;;
    K562)
      [[ "${cell_n}" == "k562" ]]
      ;;
    *)
      return 0
      ;;
  esac
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  if [[ ${#WEIGHTS[@]} -gt 0 ]]; then
    WEIGHTS=("${WEIGHTS[0]}")
  fi
  BENCHMARKS=("${BENCHMARKS[0]}")
  echo "SMOKE=1: running ${#WEIGHTS[@]} weight(s) × ${#BENCHMARKS[@]} benchmark(s)"
  echo ""
fi

SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_KEYS_FILE="${OUT_DIR}/.skip_existing_keys.txt"
: > "${SKIP_KEYS_FILE}"
if [[ "${SKIP_EXISTING}" == "1" && -f "${RESULTS_JSONL}" ]]; then
  python - "${RESULTS_JSONL}" "${SKIP_KEYS_FILE}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

jsonl, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
from eval_split_args import eval_result_cell_key_from_record

keys = []
for line in jsonl.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    record = json.loads(line)
    if record.get("status") != "ok":
        continue
    keys.append(eval_result_cell_key_from_record(record))
out.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
print(f"SKIP_EXISTING=1: {len(keys)} completed cell(s)")
PY
fi

if [[ "${ENSEMBLE_ONLY:-0}" == "1" ]]; then
  WANT_PRIDICT2_ENSEMBLE=1
fi

TOTAL=$((${#WEIGHTS[@]} * ${#BENCHMARKS[@]}))
ENS_HEADS=("${PRIDICT2_HEAD_ARR[@]}")
ENS_RUNS=(0 1 2 3 4)
if [[ "${SMOKE:-0}" == "1" ]]; then
  ENS_HEADS=("${ENS_HEADS[0]}")
  ENS_RUNS=(0)
fi
ENS_TOTAL=0
if [[ "${WANT_PRIDICT2_ENSEMBLE}" == "1" ]]; then
  if [[ "${MATCH_PRIDICT2_HEAD_TO_CELL}" == "1" ]]; then
    for head in "${ENS_HEADS[@]}"; do
      for _run in "${ENS_RUNS[@]}"; do
        for bench_spec in "${BENCHMARKS[@]}"; do
          IFS='|' read -r _n _s _d cell _p <<< "${bench_spec}"
          if pridict2_head_matches_cell "${head}" "${cell}"; then
            ENS_TOTAL=$((ENS_TOTAL + 1))
          fi
        done
      done
    done
  else
    ENS_TOTAL=$(( ${#ENS_HEADS[@]} * ${#ENS_RUNS[@]} * ${#BENCHMARKS[@]} ))
  fi
fi
echo "Vendor evaluate: ${#WEIGHTS[@]} weights × ${#BENCHMARKS[@]} benchmarks = ${TOTAL} jobs (deepprime, oped, optiprime)"
if [[ "${WANT_PRIDICT2_ENSEMBLE}" == "1" ]]; then
  echo "PRIDICT2 ensembles after that: ${ENS_TOTAL} jobs"
else
  echo "PRIDICT2 ensembles: skipped (MODELS filter)"
fi
echo "PRIDICT2_HEADS: ${PRIDICT2_HEADS_KEY}"
echo ""

cat > "${OUT_DIR}/matrix.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "device": "${DEVICE}",
  "allow_data_leak": $([ "${ALLOW_DATA_LEAK}" = "1" ] && echo true || echo false),
  "design_ruleset": $([ -n "${DESIGN_RULESET}" ] && printf '"%s"' "${DESIGN_RULESET}" || echo null),
  "n_weights": ${#WEIGHTS[@]},
  "n_benchmarks": ${#BENCHMARKS[@]},
  "n_evaluations": ${TOTAL},
  "n_ensembles": ${ENS_TOTAL}
}
EOF

IDX=0
if [[ "${ENSEMBLE_ONLY:-0}" == "1" ]]; then
  echo "ENSEMBLE_ONLY=1: skipping peen evaluate matrix"
  echo ""
elif [[ ${#WEIGHTS[@]} -eq 0 ]]; then
  echo "No vendor evaluate jobs (PRIDICT2 is ensemble-only)"
  echo ""
else
for weight_spec in "${WEIGHTS[@]}"; do
  IFS='|' read -r MODEL WEIGHTS_ID EXPERIMENT_ID CV_RUN <<< "${weight_spec}"
  for bench_spec in "${BENCHMARKS[@]}"; do
    IFS='|' read -r BENCH_NAME STUDY DATASETS_CSV CELL_LINE PE_SYSTEM <<< "${bench_spec}"
    IDX=$((IDX + 1))
    EFFECTIVE_WEIGHTS="${WEIGHTS_ID}"
    # DeepPrime ClinVar / DeepPE: author original_fold=-1 is the permanent test.
    # PRIDICT2 × library-diverse: vendor run_x held out testset_fold==x
    # (~20% of loci). Using the default test value -1 would send every
    # 0..4 row to train. Other benches: random group holdout.
    SPLIT_PLAN_JSON="$(
      python "${SCRIPT_DIR}/eval_split_args.py" --json \
        --model "${MODEL}" \
        --study "${STUDY}" \
        --datasets "${DATASETS_CSV}" \
        --cv-run "${CV_RUN}"
    )"
    SPLIT_ARGS=()
    while IFS= read -r token; do
      [[ -n "${token}" ]] && SPLIT_ARGS+=("${token}")
    done < <(python -c "import json,sys; print('\\n'.join(json.loads(sys.argv[1])['args']))" "${SPLIT_PLAN_JSON}")

    SKIP_KEY="$(
      python - "${SCRIPT_DIR}" "${SPLIT_PLAN_JSON}" "${MODEL}" "${EFFECTIVE_WEIGHTS}" "${BENCH_NAME}" "${CELL_LINE}" "${PE_SYSTEM}" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from eval_split_args import eval_result_cell_key
plan = json.loads(sys.argv[2])
print(eval_result_cell_key(
    model=sys.argv[3],
    weights=sys.argv[4],
    benchmark_name=sys.argv[5],
    cell_line=sys.argv[6],
    original_fold_test_value=plan["original_fold_test_value"],
    pe_system=sys.argv[7] or None,
))
PY
    )"
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      if grep -Fqx "${SKIP_KEY}" "${SKIP_KEYS_FILE}"; then
        echo "[${IDX}/${TOTAL}] skip ${MODEL} / ${EFFECTIVE_WEIGHTS} @ ${BENCH_NAME}"
        continue
      fi
    fi
    FOLD_TAG="$(python -c "import json,sys; v=json.loads(sys.argv[1]).get('original_fold_test_value'); print('' if v is None else f'__fold_{int(v)}')" "${SPLIT_PLAN_JSON}")"
    SAFE_NAME="$(echo "${MODEL}__${EFFECTIVE_WEIGHTS}__${BENCH_NAME}${FOLD_TAG}" | tr '/:' '__')"
    STDOUT_FILE="${LOG_DIR}/${SAFE_NAME}.stdout"
    STDERR_FILE="${LOG_DIR}/${SAFE_NAME}.stderr"

    echo "[${IDX}/${TOTAL}] ${MODEL} / ${EFFECTIVE_WEIGHTS} @ ${BENCH_NAME}"

    DATASET_ARGS=()
    IFS=',' read -ra DS_ARR <<< "${DATASETS_CSV}"
    for ds in "${DS_ARR[@]}"; do
      DATASET_ARGS+=(--dataset "${ds}")
    done
    if [[ -n "${CELL_LINE}" ]]; then
      DATASET_ARGS+=(--cell-line "${CELL_LINE}")
    fi
    if [[ -n "${PE_SYSTEM}" ]]; then
      DATASET_ARGS+=(--pe-system "${PE_SYSTEM}")
    fi

    META_JSON="$(
      python - "${SPLIT_PLAN_JSON}" "${MODEL}" "${EFFECTIVE_WEIGHTS}" "${EXPERIMENT_ID}" "${CV_RUN}" "${STUDY}" "${DATASETS_CSV}" "${CELL_LINE}" "${BENCH_NAME}" "${PE_SYSTEM}" "${ALLOW_DATA_LEAK}" "${DESIGN_RULESET}" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
cv_run = sys.argv[5].strip()
print(json.dumps({
    "model": sys.argv[2],
    "weights": sys.argv[3],
    "experiment_id": sys.argv[4] or None,
    "cv_run": int(cv_run) if cv_run else None,
    "study": sys.argv[6],
    "datasets": [item for item in sys.argv[7].split(",") if item],
    "cell_line": sys.argv[8] or None,
    "benchmark_name": sys.argv[9],
    "pe_system": sys.argv[10] or None,
    "use_original_fold": plan["use_original_fold"],
    "original_fold_test_value": plan["original_fold_test_value"],
    "allow_data_leak": sys.argv[11] == "1",
    "design_ruleset": sys.argv[12] or None,
}))
PY
    )"

    set +e
    "${PEEN_CMD[@]}" evaluate \
      --model "${MODEL}" \
      --weights "${EFFECTIVE_WEIGHTS}" \
      --custom-benchmark \
      --benchmark-name "${BENCH_NAME}" \
      --study "${STUDY}" \
      "${DATASET_ARGS[@]}" \
      "${DESIGN_RULE_ARGS[@]}" \
      "${SPLIT_ARGS[@]}" \
      "${LEAK_ARGS[@]}" \
      --sync \
      --device "${DEVICE}" \
      > "${STDOUT_FILE}" 2> "${STDERR_FILE}"
    EXIT_CODE=$?
    set -e

    python - "${RESULTS_JSONL}" "${STDOUT_FILE}" "${STDERR_FILE}" "${EXIT_CODE}" "${META_JSON}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
stdout_path = Path(sys.argv[2])
stderr_path = Path(sys.argv[3])
exit_code = int(sys.argv[4])
meta = json.loads(sys.argv[5])
sys.path.insert(0, sys.argv[6])
from summarize_eval_results import extract_json_object

stdout = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()

record = dict(meta)
record["exit_code"] = exit_code

payload = extract_json_object(stdout) if stdout else None

if payload is not None:
    record.update(payload)
    if payload.get("skipped"):
        record.setdefault("status", "skipped")
    elif payload.get("error_type") == "data_leak" or payload.get("status") == "error":
        record.setdefault("status", "error")
    elif payload.get("metrics") is not None:
        record.setdefault("status", "ok")
    else:
        record.setdefault("status", payload.get("status") or "unknown")
else:
    record["status"] = "error"
    record["error_type"] = "cli_failure"
    record["metrics"] = None
    record["n_samples"] = None
    record["stderr_tail"] = stderr[-2000:] if stderr else None
    if not record.get("model"):
        record["model"] = None
    if not record.get("weights"):
        record["weights"] = None

with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, default=str) + "\n")
PY

  done
done
fi

# ---------------------------------------------------------------------------
# PRIDICT2 ensemble evaluation: mean of pridict1_1 + pridict1_2 per CV fold
# ---------------------------------------------------------------------------
ENSEMBLE_MODEL="pridict2"
if [[ "${WANT_PRIDICT2_ENSEMBLE}" == "1" ]]; then
  ENS_IDX=0
  echo ""
  echo "=== PRIDICT2 ensemble (mean of pridict1_1 + pridict1_2) ==="
  echo "Matrix: ${#ENS_HEADS[@]} heads × ${#ENS_RUNS[@]} folds × ${#BENCHMARKS[@]} benchmarks = ${ENS_TOTAL} ensembles"
  echo ""

  for head in "${ENS_HEADS[@]}"; do
    for run in "${ENS_RUNS[@]}"; do
      MEMBER_A="${PRIDICT2_EXPERIMENTS[0]}__run_${run}__${head}"
      MEMBER_B="${PRIDICT2_EXPERIMENTS[1]}__run_${run}__${head}"
      ENS_WEIGHTS="ensemble__run_${run}__${head}"
      ENS_EXPERIMENT_ID="pridict2_ensemble"

      for bench_spec in "${BENCHMARKS[@]}"; do
        IFS='|' read -r BENCH_NAME STUDY DATASETS_CSV CELL_LINE PE_SYSTEM <<< "${bench_spec}"
        ENS_NAME="pridict2-ensemble-${head}-run${run}-${BENCH_NAME}"
        if [[ "${MATCH_PRIDICT2_HEAD_TO_CELL}" == "1" ]] && ! pridict2_head_matches_cell "${head}" "${CELL_LINE}"; then
          continue
        fi
        ENS_IDX=$((ENS_IDX + 1))

        SPLIT_PLAN_JSON="$(
          python "${SCRIPT_DIR}/eval_split_args.py" --json \
            --model "${ENSEMBLE_MODEL}" \
            --study "${STUDY}" \
            --datasets "${DATASETS_CSV}" \
            --cv-run "${run}"
        )"
        SPLIT_ARGS=()
        while IFS= read -r token; do
          [[ -n "${token}" ]] && SPLIT_ARGS+=("${token}")
        done < <(python -c "import json,sys; print('\\n'.join(json.loads(sys.argv[1])['args']))" "${SPLIT_PLAN_JSON}")

        SKIP_KEY="$(
          python - "${SCRIPT_DIR}" "${SPLIT_PLAN_JSON}" "${ENSEMBLE_MODEL}" "${ENS_WEIGHTS}" "${BENCH_NAME}" "${CELL_LINE}" "${PE_SYSTEM}" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from eval_split_args import eval_result_cell_key
plan = json.loads(sys.argv[2])
print(eval_result_cell_key(
    model=sys.argv[3],
    weights=sys.argv[4],
    benchmark_name=sys.argv[5],
    cell_line=sys.argv[6],
    original_fold_test_value=plan["original_fold_test_value"],
    pe_system=sys.argv[7] or None,
))
PY
        )"
        if [[ "${SKIP_EXISTING}" == "1" ]]; then
          if grep -Fqx "${SKIP_KEY}" "${SKIP_KEYS_FILE}"; then
            echo "[ens ${ENS_IDX}/${ENS_TOTAL}] skip ${ENS_NAME}"
            continue
          fi
        fi

        FOLD_TAG="$(python -c "import json,sys; v=json.loads(sys.argv[1]).get('original_fold_test_value'); print('' if v is None else f'__fold_{int(v)}')" "${SPLIT_PLAN_JSON}")"
        SAFE_NAME="$(echo "${ENSEMBLE_MODEL}__${ENS_WEIGHTS}__${BENCH_NAME}${FOLD_TAG}" | tr '/:' '__')"
        STDOUT_FILE="${LOG_DIR}/${SAFE_NAME}.stdout"
        STDERR_FILE="${LOG_DIR}/${SAFE_NAME}.stderr"

        echo "[ens ${ENS_IDX}/${ENS_TOTAL}] ${ENS_NAME}"

        DATASET_ARGS=()
        IFS=',' read -ra DS_ARR <<< "${DATASETS_CSV}"
        for ds in "${DS_ARR[@]}"; do
          DATASET_ARGS+=(--dataset "${ds}")
        done
        if [[ -n "${CELL_LINE}" ]]; then
          DATASET_ARGS+=(--cell-line "${CELL_LINE}")
        fi
        if [[ -n "${PE_SYSTEM}" ]]; then
          DATASET_ARGS+=(--pe-system "${PE_SYSTEM}")
        fi

        META_JSON="$(
          python - "${SPLIT_PLAN_JSON}" "${ENSEMBLE_MODEL}" "${ENS_WEIGHTS}" "${ENS_EXPERIMENT_ID}" "${run}" "${STUDY}" "${DATASETS_CSV}" "${CELL_LINE}" "${BENCH_NAME}" "${MEMBER_A}" "${MEMBER_B}" "${PE_SYSTEM}" "${ALLOW_DATA_LEAK}" "${DESIGN_RULESET}" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
cv_run = sys.argv[5].strip()
print(json.dumps({
    "model": sys.argv[2],
    "weights": sys.argv[3],
    "experiment_id": sys.argv[4] or None,
    "cv_run": int(cv_run) if cv_run else None,
    "study": sys.argv[6],
    "datasets": [item for item in sys.argv[7].split(",") if item],
    "cell_line": sys.argv[8] or None,
    "benchmark_name": sys.argv[9],
    "pe_system": sys.argv[12] or None,
    "use_original_fold": plan["use_original_fold"],
    "original_fold_test_value": plan["original_fold_test_value"],
    "ensemble": True,
    "ensemble_members": [sys.argv[10], sys.argv[11]],
    "allow_data_leak": sys.argv[13] == "1",
    "design_ruleset": sys.argv[14] or None,
}))
PY
        )"

        set +e
        "${PEEN_CMD[@]}" ensemble \
          --ensemble-name "${ENS_NAME}" \
          --combine mean \
          --member "${ENSEMBLE_MODEL}:${MEMBER_A}" \
          --member "${ENSEMBLE_MODEL}:${MEMBER_B}" \
          --study "${STUDY}" \
          "${DATASET_ARGS[@]}" \
          "${DESIGN_RULE_ARGS[@]}" \
          "${SPLIT_ARGS[@]}" \
          "${LEAK_ARGS[@]}" \
          --sync \
          --device "${DEVICE}" \
          > "${STDOUT_FILE}" 2> "${STDERR_FILE}"
        EXIT_CODE=$?
        set -e

        python - "${RESULTS_JSONL}" "${STDOUT_FILE}" "${STDERR_FILE}" "${EXIT_CODE}" "${META_JSON}" "${SCRIPT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
stdout_path = Path(sys.argv[2])
stderr_path = Path(sys.argv[3])
exit_code = int(sys.argv[4])
meta = json.loads(sys.argv[5])
sys.path.insert(0, sys.argv[6])
from summarize_eval_results import extract_json_object

stdout = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()

record = dict(meta)
record["exit_code"] = exit_code

payload = extract_json_object(stdout) if stdout else None

if payload is not None:
    # Keep meta identity fields; ensemble JSON has no model/weights/benchmark_name.
    for key, value in payload.items():
        if key in (
            "model",
            "weights",
            "experiment_id",
            "cv_run",
            "study",
            "datasets",
            "cell_line",
            "pe_system",
            "benchmark_name",
            "use_original_fold",
            "original_fold_test_value",
            "ensemble",
            "ensemble_members",
        ):
            continue
        record[key] = value
    if payload.get("skipped"):
        record.setdefault("status", "skipped")
    elif payload.get("error_type") == "data_leak" or payload.get("status") == "error":
        record.setdefault("status", "error")
    elif payload.get("metrics") is not None:
        record.setdefault("status", "ok")
    else:
        record.setdefault("status", payload.get("status") or "unknown")
else:
    record["status"] = "error"
    record["error_type"] = "cli_failure"
    record["metrics"] = None
    record["n_samples"] = None
    record["stderr_tail"] = stderr[-2000:] if stderr else None

with out_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, default=str) + "\n")
PY

      done
    done
  done
fi

echo ""
echo "Wrote ${RESULTS_JSONL}"
python - "${RESULTS_JSONL}" "${SCRIPT_DIR}" <<'PY'
"""Keep the last record per eval cell (model, weights, benchmark, cell, pe, fold)."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from eval_split_args import eval_result_cell_key_from_record

if not path.is_file():
    raise SystemExit(0)
records = []
index = {}
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    record = json.loads(line)
    key = eval_result_cell_key_from_record(record)
    if key in index:
        records[index[key]] = record
    else:
        index[key] = len(records)
        records.append(record)
path.write_text(
    "".join(json.dumps(record, default=str) + "\n" for record in records),
    encoding="utf-8",
)
print(f"Compacted {path} to {len(records)} unique cells")
PY
echo "Summarizing..."
python "${SCRIPT_DIR}/summarize_eval_results.py" "${RESULTS_JSONL}"
printf '%s\n' "${RUN_ID}" > "${OUT_ROOT}/LATEST_RUN_ID"
echo "Done. Results under ${OUT_DIR}"

#!/usr/bin/env bash
# Evaluate base vendor weights on pooled PE-DB benchmarks via peen evaluate --sync.
#
# Weights:
#   - deepprime / DeepPrime_base
#   - oped / pegRNA_Model_Merged_saved.order3_decoder_weights
#   - optiprime / base
#   - pridict2 / August 2023 CV folds × HEK and K562 heads
#
# Benchmarks (dataset lists; multi-cell-line datasets are expanded per cell):
#   minsepie-insert-pooled, deeppe-pooled, deepprime-clinvar,
#   pridict1-library1, pridict2-library-diverse, optiprime-lib-mmr, optiprime-lib-cv
#
# Splits:
#   - DeepPrime ClinVar: author original_fold=-1 test
#   - PRIDICT2 × library-diverse: run_x tests on author testset_fold==x
#   - PRIDICT library1 has no author split: vendor training used every locus
#   - everything else: random group holdout
#
# Usage:
#   conda activate pedb
#   DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
#
# Env:
#   DEVICE            compute device (default: auto)
#   RUN_ID            output run id (default: UTC timestamp). Reuse an existing id
#                     to overwrite matching cells and refresh summary.csv.
#   OUT_ROOT          results root (default: <repo>/results/base_model_eval)
#   MODELS            comma/space list to restrict weights (deepprime,oped,optiprime,pridict2)
#   BENCHMARKS        comma/space list of benchmark names
#   PRIDICT2_HEADS    comma/space list of cell-type heads (default: HEK,K562).
#                     Each PRIDICT2 CV run is scored with every listed head on
#                     every benchmark (cross-cell, not just the matching head).
#   SKIP_EXISTING=1   skip cells already ok in results.jsonl
#                     (library-diverse fold-matched cells are distinct from
#                     earlier random-holdout rows of the same weight)
#   SMOKE=1           evaluate only first remaining weight × first remaining benchmark
#
# Partial rerun into an existing run (DeepPrime numbers stay; others refresh):
#   DEVICE=cuda:0 MODELS=oped,pridict2,optiprime RUN_ID=20260902T151810Z \
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
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/base_model_eval}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
RESULTS_JSONL="${OUT_DIR}/results.jsonl"
LOG_DIR="${OUT_DIR}/logs"
FILTER_MODELS="${MODELS:-}"
FILTER_BENCHMARKS="${BENCHMARKS:-}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

print_experiment_banner "Base model evaluation (pooled benchmarks)"
echo "RUN_ID:    ${RUN_ID}"
echo "OUT_DIR:   ${OUT_DIR}"
echo "DEVICE:    ${DEVICE}"
echo "PEEN_CMD:  ${PEEN_CMD[*]}"
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
# Multi-head vendor runs require a cell-type suffix. Score every listed head
# on every benchmark (HEK and K562 by default) so HEK-trained vs K562-trained
# decoders can be compared on the same sheet. CV mean/std is over run_0..4
# within each head.
PRIDICT2_HEADS_KEY="$(echo "${PRIDICT2_HEADS:-HEK,K562}" | tr -s ' ,' ',' | sed 's/^,//;s/,$//')"
IFS=',' read -ra PRIDICT2_HEAD_ARR <<< "${PRIDICT2_HEADS_KEY}"
if [[ ${#PRIDICT2_HEAD_ARR[@]} -eq 0 ]]; then
  echo "Error: PRIDICT2_HEADS='${PRIDICT2_HEADS:-}' produced no heads" >&2
  exit 1
fi
for exp in "${PRIDICT2_EXPERIMENTS[@]}"; do
  for run in 0 1 2 3 4; do
    for head in "${PRIDICT2_HEAD_ARR[@]}"; do
      WEIGHTS+=("pridict2|${exp}__run_${run}__${head}|${exp}|${run}")
    done
  done
done

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
  if [[ ${#WEIGHTS[@]} -eq 0 ]]; then
    echo "Error: MODELS='${FILTER_MODELS}' matched no weights (deepprime|oped|optiprime|pridict2)" >&2
    exit 1
  fi
fi

if [[ -n "${FILTER_BENCHMARKS}" ]]; then
  BENCH_KEY=",$(echo "${FILTER_BENCHMARKS}" | tr -s ' ,' ',' | sed 's/^,//;s/,$//'),"
  FILTERED_BENCHES=()
  for spec in "${BENCHMARKS[@]}"; do
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

# Split multi-cell-line datasets (library-diverse HEK/K562/K562MLH1dn, etc.)
mapfile -t BENCHMARKS < <(
  python "${SCRIPT_DIR}/expand_eval_cell_lines.py" \
    --datasets-dir "${REPO_ROOT}/datasets" \
    "${BENCHMARKS[@]}"
)
if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  echo "Error: no benchmarks after cell-line expansion" >&2
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  WEIGHTS=("${WEIGHTS[0]}")
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

TOTAL=$((${#WEIGHTS[@]} * ${#BENCHMARKS[@]}))
echo "Matrix: ${#WEIGHTS[@]} weights × ${#BENCHMARKS[@]} benchmarks = ${TOTAL} evaluations"
echo "PRIDICT2_HEADS: ${PRIDICT2_HEADS_KEY}"
echo ""

cat > "${OUT_DIR}/matrix.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "device": "${DEVICE}",
  "n_weights": ${#WEIGHTS[@]},
  "n_benchmarks": ${#BENCHMARKS[@]},
  "n_evaluations": ${TOTAL}
}
EOF

IDX=0
for weight_spec in "${WEIGHTS[@]}"; do
  IFS='|' read -r MODEL WEIGHTS_ID EXPERIMENT_ID CV_RUN <<< "${weight_spec}"
  for bench_spec in "${BENCHMARKS[@]}"; do
    IFS='|' read -r BENCH_NAME STUDY DATASETS_CSV CELL_LINE <<< "${bench_spec}"
    IDX=$((IDX + 1))
    EFFECTIVE_WEIGHTS="${WEIGHTS_ID}"
    # DeepPrime ClinVar: author original_fold=-1 is the permanent test set.
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
      python - "${SCRIPT_DIR}" "${SPLIT_PLAN_JSON}" "${MODEL}" "${EFFECTIVE_WEIGHTS}" "${BENCH_NAME}" "${CELL_LINE}" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from eval_split_args import eval_result_cell_key
plan = json.loads(sys.argv[2])
print(eval_result_cell_key(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], plan["original_fold_test_value"]))
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

    META_JSON="$(
      python - "${SPLIT_PLAN_JSON}" "${MODEL}" "${EFFECTIVE_WEIGHTS}" "${EXPERIMENT_ID}" "${CV_RUN}" "${STUDY}" "${DATASETS_CSV}" "${CELL_LINE}" "${BENCH_NAME}" <<'PY'
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
    "use_original_fold": plan["use_original_fold"],
    "original_fold_test_value": plan["original_fold_test_value"],
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
      "${SPLIT_ARGS[@]}" \
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

echo ""
echo "Wrote ${RESULTS_JSONL}"
python - "${RESULTS_JSONL}" "${SCRIPT_DIR}" <<'PY'
"""Keep the last record per eval cell (model, weights, benchmark, cell, fold)."""
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

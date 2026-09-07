#!/usr/bin/env bash
#
# Run PE-Hub tests in isolated pytest processes.
#
# pe-db lives under ``pe_db`` and pe-ensemble under ``pe_ensemble``. Each suite
# gets its own interpreter and PYTHONPATH so imports resolve without an
# editable install and plugin state stays per suite.
#
# Usage:
#   ./scripts/run-tests.sh                 # every suite
#   ./scripts/run-tests.sh --list          # named groups
#   ./scripts/run-tests.sh smoke           # one functional group
#   ./scripts/run-tests.sh training jobs   # union of groups
#   ./scripts/run-tests.sh --install
#   ./scripts/run-tests.sh smoke -v        # extra pytest args
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMON="${REPO_ROOT}/packages/pe-common"

# shellcheck source=scripts/python-env.sh
source "${REPO_ROOT}/scripts/python-env.sh"

if ! PYTHON="$(pe_hub_resolve_python)"; then
    echo "Error: python3 is not installed" >&2
    exit 1
fi
if ! pe_hub_require_python_version "${PYTHON}" >/dev/null 2>&1; then
    for candidate in \
        "${HOME}/miniconda3/envs/pe-hub/bin/python" \
        "${HOME}/anaconda3/envs/pe-hub/bin/python" \
        "${REPO_ROOT}/venv/bin/python"
    do
        if [[ -x "${candidate}" ]] && pe_hub_require_python_version "${candidate}" >/dev/null 2>&1; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if ! pe_hub_require_python_version "${PYTHON}"; then
    exit 1
fi

INSTALL_DEPS=false
LIST_ONLY=false
CHECK_GROUPS=false
SELECTED_GROUPS=()
PYTEST_ARGS=()

# ---------------------------------------------------------------------------
# Suites: isolated pytest processes. "label|tests_path|extra_pythonpath"
# ---------------------------------------------------------------------------
SUITES=(
    "pe-common|${COMMON}/tests|${COMMON}"
    "pe-common-lib|${REPO_ROOT}/packages/tests|${COMMON}"
    "pe-db|${REPO_ROOT}/services/pe-db/tests|${REPO_ROOT}/services/pe-db:${COMMON}"
    "pe-ensemble|${REPO_ROOT}/services/pe-ensemble/tests|${REPO_ROOT}/services/pe-ensemble:${COMMON}"
    "experiments|${REPO_ROOT}/scripts/experiments/datasheet-benchmark/test_protocol.py|${REPO_ROOT}/scripts/experiments/datasheet-benchmark:${COMMON}"
)

# ---------------------------------------------------------------------------
# Functional groups: relative paths (and optional pytest node ids).
# Suite aliases expand to every test_*.py under that suite.
# ---------------------------------------------------------------------------
GROUP_NAMES=(
    all
    smoke
    integration
    pe-common
    pe-db
    pe-ensemble
    experiments
    data
    conversion
    plugins
    models
    training
    evaluation
    jobs
    splits
    utils
)

group_description() {
    case "$1" in
        all) echo "Every test suite (default)" ;;
        smoke) echo "HTTP and CLI operation-path tests" ;;
        integration) echo "Slow tests that load vendor models or full data" ;;
        pe-common) echo "Shared library tests (packages/pe-common + packages/tests)" ;;
        pe-db) echo "PE Database catalog, conversion, and API tests" ;;
        pe-ensemble) echo "Model, training, evaluation, and ensemble tests" ;;
        experiments) echo "Experiment-script unit tests (datasheet-benchmark protocol)" ;;
        data) echo "Catalog, library filter, statistics, target UIDs, progress logs" ;;
        conversion) echo "Standardized <-> model-format conversion and cache" ;;
        plugins) echo "Plugin upload, validation, activation, and loaders" ;;
        models) echo "Wrappers, weights, encodings, vendor evaluate()" ;;
        training) echo "Training, tuning, presets, and architecture" ;;
        evaluation) echo "Evaluation, leakage, ensemble fusion, and benchmarks" ;;
        jobs) echo "Job registries, scheduler, and cancel/kill" ;;
        splits) echo "Split assignment and eval-split helpers" ;;
        utils) echo "Shared pe-common helpers (devices, sequences, cell lines, filter params)" ;;
        *) echo "" ;;
    esac
}

_suite_test_files() {
    local tests_path="$1"
    if [[ -d "${tests_path}" ]]; then
        find "${tests_path}" -name 'test_*.py' -type f | sort
    elif [[ -f "${tests_path}" ]]; then
        printf '%s\n' "${tests_path}"
    fi
}

_relpath() {
    local path="$1"
    path="${path#"${REPO_ROOT}/"}"
    printf '%s\n' "${path}"
}

group_entries() {
    # Prints repo-relative test paths or node ids, one per line.
    case "$1" in
        all)
            for suite in "${SUITES[@]}"; do
                IFS="|" read -r _label tests_path _ <<< "${suite}"
                while IFS= read -r path; do
                    _relpath "${path}"
                done < <(_suite_test_files "${tests_path}")
            done
            ;;
        pe-common)
            group_entries all | grep -E '^packages/(pe-common/tests|tests)/' || true
            ;;
        pe-db)
            group_entries all | grep -E '^services/pe-db/tests/' || true
            ;;
        pe-ensemble)
            group_entries all | grep -E '^services/pe-ensemble/tests/' || true
            ;;
        experiments)
            echo "scripts/experiments/datasheet-benchmark/test_protocol.py"
            ;;
        smoke)
            cat <<'EOF'
services/pe-db/tests/test_api_smoke.py
services/pe-db/tests/test_cli_smoke.py
services/pe-ensemble/tests/test_api_smoke.py
services/pe-ensemble/tests/test_cli_smoke.py
services/pe-ensemble/tests/test_pe_ensemble_cli.py
EOF
            ;;
        integration)
            cat <<'EOF'
services/pe-ensemble/tests/test_vendor_models_evaluation.py
services/pe-ensemble/tests/test_model_wrappers.py::TestIntegration
EOF
            ;;
        data)
            cat <<'EOF'
packages/pe-common/tests/test_conversion_progress.py
packages/tests/test_target_uid.py
services/pe-db/tests/test_library.py
services/pe-db/tests/test_statistics.py
services/pe-ensemble/tests/test_pe_db_library.py
services/pe-ensemble/tests/test_conversion_progress.py
services/pe-ensemble/tests/test_progress_log.py
EOF
            ;;
        conversion)
            cat <<'EOF'
services/pe-db/tests/test_conversions.py
services/pe-db/tests/test_formatted_cache.py
services/pe-db/tests/test_window_geometry.py
services/pe-db/tests/test_mfe_process_pool.py
services/pe-db/tests/test_deepspcas9.py
services/pe-db/tests/test_pipeline_registry.py
EOF
            ;;
        plugins)
            cat <<'EOF'
packages/pe-common/tests/test_plugins.py
packages/pe-common/tests/test_plugin_validation.py
services/pe-db/tests/test_plugin_loader.py
services/pe-db/tests/test_plugin_reload.py
services/pe-ensemble/tests/test_plugin_manager.py
services/pe-ensemble/tests/test_plugin_validation_jobs.py
services/pe-ensemble/tests/test_ensemble_plugin_loader.py
services/pe-ensemble/tests/test_train_models_plugins.py
EOF
            ;;
        models)
            cat <<'EOF'
services/pe-ensemble/tests/test_model_wrappers.py
services/pe-ensemble/tests/test_model_architecture.py
services/pe-ensemble/tests/test_hparams.py
services/pe-ensemble/tests/test_weights_loading.py
services/pe-ensemble/tests/test_weights_registry.py
services/pe-ensemble/tests/test_vendor_models_evaluation.py
services/pe-ensemble/tests/test_optiprime_wrapper.py
services/pe-ensemble/tests/test_oped_encoding.py
services/pe-ensemble/tests/test_pridict2_align_seqs.py
services/pe-ensemble/tests/test_pridict2_batch_seq_ids.py
services/pe-ensemble/tests/test_pridict2_lightning.py
services/pe-ensemble/tests/test_pridict2_perbase_dtypes.py
services/pe-ensemble/tests/test_pridict2_vendor_provenance.py
services/pe-ensemble/tests/test_pridict2_weight_bundles.py
services/pe-ensemble/tests/test_pridict2_weight_selection.py
services/pe-ensemble/tests/test_author_folds.py
EOF
            ;;
        training)
            cat <<'EOF'
packages/tests/test_training.py
services/pe-ensemble/tests/test_training_jobs.py
services/pe-ensemble/tests/test_training_reproducibility.py
services/pe-ensemble/tests/test_tune_jobs.py
services/pe-ensemble/tests/test_tune_study.py
services/pe-ensemble/tests/test_hyperparameter_presets.py
services/pe-ensemble/tests/test_model_architecture.py
services/pe-ensemble/tests/test_train_models_plugins.py
EOF
            ;;
        evaluation)
            cat <<'EOF'
services/pe-ensemble/tests/test_evaluation_jobs.py
services/pe-ensemble/tests/test_evaluation_leakage.py
services/pe-ensemble/tests/test_evaluation_skip.py
services/pe-ensemble/tests/test_evaluation_validation.py
services/pe-ensemble/tests/test_evaluation_benchmark.py
services/pe-ensemble/tests/test_eval_split_args.py
services/pe-ensemble/tests/test_expand_eval_cell_lines.py
services/pe-ensemble/tests/test_summarize_eval_results.py
services/pe-ensemble/tests/test_ensemble_combine.py
services/pe-ensemble/tests/test_ensemble_align.py
services/pe-ensemble/tests/test_vendor_models_evaluation.py
EOF
            ;;
        jobs)
            cat <<'EOF'
services/pe-ensemble/tests/test_training_jobs.py
services/pe-ensemble/tests/test_tune_jobs.py
services/pe-ensemble/tests/test_evaluation_jobs.py
services/pe-ensemble/tests/test_plugin_validation_jobs.py
services/pe-ensemble/tests/test_job_kill.py
services/pe-ensemble/tests/test_device_scheduler.py
services/pe-ensemble/tests/test_progress_log.py
EOF
            ;;
        splits)
            cat <<'EOF'
packages/tests/test_splits.py
services/pe-ensemble/tests/test_split_request_validation.py
services/pe-ensemble/tests/test_eval_split_args.py
scripts/experiments/datasheet-benchmark/test_protocol.py
EOF
            ;;
        utils)
            cat <<'EOF'
packages/pe-common/tests/test_devices.py
packages/pe-common/tests/test_cell_lines.py
packages/pe-common/tests/test_sequence_utils.py
packages/pe-common/tests/test_conversion_progress.py
packages/pe-common/tests/test_filter_params.py
EOF
            ;;
        *)
            return 1
            ;;
    esac
}

is_known_group() {
    local name="$1"
    local group
    for group in "${GROUP_NAMES[@]}"; do
        if [[ "${group}" == "${name}" ]]; then
            return 0
        fi
    done
    return 1
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [GROUP ...] [-- PYTEST_ARGS...]

Run PE-Hub tests in isolated pytest processes (one process per suite).

Options:
  --install      Install pytest, httpx, and pe-common before running
  --list         List groups (with --check, also report ungrouped files)
  --check        With --list, fail if a test file is missing from every
                 functional group
  -h, --help     Show this help message

Groups (omit for all tests; multiple groups are unioned):
EOF
    local group
    for group in "${GROUP_NAMES[@]}"; do
        printf '  %-14s %s\n' "${group}" "$(group_description "${group}")"
    done
    cat <<EOF

Examples:
  $(basename "$0")
  $(basename "$0") smoke
  $(basename "$0") training evaluation
  $(basename "$0") plugins -v
  $(basename "$0") --list
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL_DEPS=true; shift ;;
        --list) LIST_ONLY=true; shift ;;
        --check) CHECK_GROUPS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS+=("$@"); break ;;
        -*) PYTEST_ARGS+=("$1"); shift ;;
        *)
            if is_known_group "$1"; then
                SELECTED_GROUPS+=("$1")
                shift
            else
                echo "Unknown test group: $1" >&2
                echo "Known groups: ${GROUP_NAMES[*]}" >&2
                echo "Use --list to see descriptions, or pass pytest args after --" >&2
                exit 2
            fi
            ;;
    esac
done

union_group_files() {
    local group
    local -A seen=()
    for group in "${SELECTED_GROUPS[@]}"; do
        while IFS= read -r entry; do
            [[ -z "${entry}" ]] && continue
            if [[ -z "${seen[${entry}]+x}" ]]; then
                seen["${entry}"]=1
                printf '%s\n' "${entry}"
            fi
        done < <(group_entries "${group}")
    done
}

list_groups() {
    local group
    echo "Test groups (run with: $0 GROUP [GROUP...])"
    echo ""
    for group in "${GROUP_NAMES[@]}"; do
        local count
        count="$(group_entries "${group}" | grep -c . || true)"
        printf '  %-14s %3s  %s\n' "${group}" "${count}" "$(group_description "${group}")"
        if [[ ${#SELECTED_GROUPS[@]} -gt 0 ]]; then
            local selected
            for selected in "${SELECTED_GROUPS[@]}"; do
                if [[ "${selected}" == "${group}" ]]; then
                    group_entries "${group}" | sed 's/^/                   /'
                fi
            done
        fi
    done
}

ungrouped_test_files() {
    local -A grouped=()
    local group entry rel
    for group in "${GROUP_NAMES[@]}"; do
        case "${group}" in
            all|pe-common|pe-db|pe-ensemble|experiments) continue ;;
        esac
        while IFS= read -r entry; do
            rel="${entry%%::*}"
            grouped["${rel}"]=1
        done < <(group_entries "${group}")
    done
    while IFS= read -r path; do
        rel="$(_relpath "${path}")"
        if [[ -z "${grouped[${rel}]+x}" ]]; then
            printf '%s\n' "${rel}"
        fi
    done < <(
        for suite in "${SUITES[@]}"; do
            IFS="|" read -r _label tests_path _ <<< "${suite}"
            _suite_test_files "${tests_path}"
        done
    )
}

if [[ "${LIST_ONLY}" == true ]]; then
    list_groups
    if [[ ${#SELECTED_GROUPS[@]} -gt 0 ]]; then
        echo ""
        echo "Selected files:"
        union_group_files | sed 's/^/  /'
    fi
fi

if [[ "${CHECK_GROUPS}" == true ]]; then
    missing="$(ungrouped_test_files || true)"
    if [[ -n "${missing}" ]]; then
        echo "" >&2
        echo "Ungrouped test files:" >&2
        while IFS= read -r path; do
            printf '  %s\n' "${path}" >&2
        done <<< "${missing}"
        exit 1
    fi
    echo ""
    echo "All test files belong to at least one functional group."
fi

if [[ "${LIST_ONLY}" == true || "${CHECK_GROUPS}" == true ]]; then
    exit 0
fi

if [[ "${INSTALL_DEPS}" == true ]]; then
    echo "Installing test dependencies (this may take a while)..."
    "${PYTHON}" -m pip install pytest httpx
    "${PYTHON}" -m pip install -e "${COMMON}"
    "${PYTHON}" -m pip install -e "${REPO_ROOT}/services/pe-ensemble"
    echo "Done."
fi

if ! "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
    echo "Error: pytest is not installed. Run: $(basename "$0") --install" >&2
    echo "       (or: ${PYTHON} -m pip install pytest)" >&2
    exit 1
fi

cd "${REPO_ROOT}"

declare -a SELECTED_FILES=()
if [[ ${#SELECTED_GROUPS[@]} -eq 0 ]]; then
    SELECTED_GROUPS=(all)
fi
if [[ "${SELECTED_GROUPS[*]}" == "all" ]]; then
    SELECTED_FILES=()
else
    mapfile -t SELECTED_FILES < <(union_group_files)
    if [[ ${#SELECTED_FILES[@]} -eq 0 ]]; then
        echo "No test files matched groups: ${SELECTED_GROUPS[*]}" >&2
        exit 1
    fi
fi

entry_in_suite() {
    local entry="$1"
    local tests_path="$2"
    local rel="${entry%%::*}"
    local abs="${REPO_ROOT}/${rel}"
    if [[ -d "${tests_path}" ]]; then
        [[ "${abs}" == "${tests_path}/"* ]] || [[ "${abs}" == "${tests_path}" ]]
    else
        [[ "${abs}" == "${tests_path}" ]] || [[ "${abs}" == "${tests_path}"* ]]
    fi
}

declare -a RESULTS=()
overall_status=0

for suite in "${SUITES[@]}"; do
    IFS="|" read -r label tests_path extra_path <<< "${suite}"

    if [[ ! -e "${tests_path}" ]]; then
        echo "▶ ${label}: no tests path (${tests_path}), skipping"
        RESULTS+=("SKIP  ${label} (no tests)")
        continue
    fi

    declare -a suite_targets=()
    if [[ "${SELECTED_GROUPS[*]}" == "all" ]]; then
        suite_targets=("${tests_path}")
    else
        for entry in "${SELECTED_FILES[@]}"; do
            if entry_in_suite "${entry}" "${tests_path}"; then
                suite_targets+=("${REPO_ROOT}/${entry}")
            fi
        done
        if [[ ${#suite_targets[@]} -eq 0 ]]; then
            continue
        fi
    fi

    echo ""
    echo "=================================================================="
    echo "▶ Running ${label} tests"
    if [[ "${SELECTED_GROUPS[*]}" != "all" ]]; then
        echo "  groups: ${SELECTED_GROUPS[*]}"
    fi
    echo "=================================================================="

    PYTHONPATH="${extra_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON}" -m pytest "${suite_targets[@]}" --continue-on-collection-errors \
        ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
    code=$?

    if [[ ${code} -eq 0 || ${code} -eq 5 ]]; then
        RESULTS+=("PASS  ${label}")
    else
        RESULTS+=("FAIL  ${label} (pytest exit ${code})")
        overall_status=1
    fi
done

if [[ ${#RESULTS[@]} -eq 0 ]]; then
    echo "No suites contained the selected groups: ${SELECTED_GROUPS[*]}" >&2
    exit 1
fi

echo ""
echo "=================================================================="
echo "Test summary"
echo "=================================================================="
for line in "${RESULTS[@]}"; do
    echo "  ${line}"
done

if [[ ${overall_status} -eq 0 ]]; then
    echo ""
    echo "All selected test suites passed."
else
    echo ""
    echo "Some test suites failed." >&2
fi

exit ${overall_status}

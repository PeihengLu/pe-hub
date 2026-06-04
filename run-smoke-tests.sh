#!/usr/bin/env bash
#
# Run all PE smoke tests across services.
#
# Each service is tested in its own pytest process because pe-db and
# pe-ensemble both expose a top-level `app` package (running them together
# would collide in sys.modules). PYTHONPATH is set per suite so imports and
# pe_common resolve without an editable install.
#
# Usage:
#   ./run-smoke-tests.sh                 # run every smoke-test suite
#   ./run-smoke-tests.sh --install       # pip install pytest + pe-common, then run
#   ./run-smoke-tests.sh -k edit_len     # forward extra args to pytest
#   ./run-smoke-tests.sh -v              # verbose
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
COMMON="${REPO_ROOT}/packages/pe-common"

INSTALL_DEPS=false
PYTEST_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- PYTEST_ARGS...]

Run all PE smoke tests (pe-db and pe-ensemble) in isolated pytest processes.

Options:
  --install      Install pytest and the pe-common package before running
  -h, --help     Show this help message

Any other arguments are forwarded to pytest, e.g.:
  $(basename "$0") -v
  $(basename "$0") -k weights
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL_DEPS=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS+=("$@"); break ;;
        *) PYTEST_ARGS+=("$1"); shift ;;
    esac
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed" >&2
    exit 1
fi

if [[ "${INSTALL_DEPS}" == true ]]; then
    echo "Installing test dependencies (this may take a while)..."
    python3 -m pip install pytest
    python3 -m pip install -e "${COMMON}"
    # pe-ensemble's editable install brings torch/lightning so the full wrapper
    # suite can run; pe-db shares these deps.
    python3 -m pip install -e "${REPO_ROOT}/services/pe-ensemble"
    echo "Done."
fi

if ! python3 -c "import pytest" >/dev/null 2>&1; then
    echo "Error: pytest is not installed. Run: $(basename "$0") --install" >&2
    echo "       (or: python3 -m pip install pytest)" >&2
    exit 1
fi

# Suite definitions: "label|tests_path|extra_pythonpath"
SUITES=(
    "pe-db|${REPO_ROOT}/services/pe-db/tests|${REPO_ROOT}/services/pe-db:${COMMON}"
    "pe-ensemble|${REPO_ROOT}/services/pe-ensemble/tests|${REPO_ROOT}/services/pe-ensemble:${COMMON}"
)

cd "${REPO_ROOT}"

declare -a RESULTS=()
overall_status=0

for suite in "${SUITES[@]}"; do
    IFS="|" read -r label tests_path extra_path <<< "${suite}"

    if [[ ! -d "${tests_path}" ]]; then
        echo "▶ ${label}: no tests directory (${tests_path}), skipping"
        RESULTS+=("SKIP  ${label} (no tests)")
        continue
    fi

    echo ""
    echo "=================================================================="
    echo "▶ Running ${label} smoke tests"
    echo "=================================================================="

    # --continue-on-collection-errors keeps lightweight smoke tests running even
    # when an unrelated test module fails to import (e.g. an optional heavy dep
    # is missing in the current environment).
    PYTHONPATH="${extra_path}${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m pytest "${tests_path}" --continue-on-collection-errors \
        ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
    code=$?

    # pytest exit code 5 means "no tests collected" — treat as a (noisy) pass.
    if [[ ${code} -eq 0 || ${code} -eq 5 ]]; then
        RESULTS+=("PASS  ${label}")
    else
        RESULTS+=("FAIL  ${label} (pytest exit ${code})")
        overall_status=1
    fi
done

echo ""
echo "=================================================================="
echo "Smoke test summary"
echo "=================================================================="
for line in "${RESULTS[@]}"; do
    echo "  ${line}"
done

if [[ ${overall_status} -eq 0 ]]; then
    echo ""
    echo "All smoke test suites passed."
else
    echo ""
    echo "Some smoke test suites failed." >&2
fi

exit ${overall_status}

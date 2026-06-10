#!/usr/bin/env bash
#
# Start the PE Database FastAPI service.
#
# Usage:
#   ./start-pe-db.sh              # dev server with reload (default)
#   ./start-pe-db.sh --install    # install deps, then start
#   ./start-pe-db.sh --no-reload  # production-style (no auto-reload)
#   ./start-pe-db.sh --force-reexport  # re-export raw data before starting
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
SERVICE_DIR="${REPO_ROOT}/services/pe-db"

INSTALL_DEPS=false
RELOAD=true
FORCE_REEXPORT=false
HOST="${PE_DB_HOST:-0.0.0.0}"
PORT="${PE_DB_PORT:-8000}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start the PE Database API (uvicorn app.main:app).

Options:
  --install         Install Python dependencies before starting
  --no-reload       Disable uvicorn auto-reload
  --force-reexport  Re-export raw study files on startup (also re-standardizes)
  --host HOST       Bind address (default: \$PE_DB_HOST or 0.0.0.0)
  --port PORT       Listen port (default: \$PE_DB_PORT or 8000)
  -h, --help        Show this help message

Environment:
  Loads \${REPO_ROOT}/.env when present.
  Common variables: DATA_ROOT, DATABASE_URL, PE_DB_HOST, PE_DB_PORT
  Startup flags set PE_DB_FORCE_EXPORT / PE_DB_FORCE_STANDARDIZE when used.

Examples:
  ./start-pe-db.sh
  ./start-pe-db.sh --install
  ./start-pe-db.sh --force-reexport
  DATA_ROOT=/path/to/datasets ./start-pe-db.sh --port 8080
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            INSTALL_DEPS=true
            shift
            ;;
        --no-reload)
            RELOAD=false
            shift
            ;;
        --force-reexport)
            FORCE_REEXPORT=true
            shift
            ;;
        --host)
            HOST="${2:?--host requires a value}"
            shift 2
            ;;
        --port)
            PORT="${2:?--port requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export PE_PROJECT_ROOT="${PE_PROJECT_ROOT:-${REPO_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${PE_DATA_ROOT:-${REPO_ROOT}/datasets}}"

if [[ "${FORCE_REEXPORT}" == true ]]; then
    export PE_DB_FORCE_EXPORT=1
    export PE_DB_FORCE_STANDARDIZE=1
fi

# Prefer an already-active conda/venv; only auto-activate repo venv when none is active.
if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "${REPO_ROOT}/venv/bin" ]]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/venv/bin/activate"
    elif [[ -d "${REPO_ROOT}/.venv/bin" ]]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/.venv/bin/activate"
    fi
fi

resolve_python() {
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        return 1
    fi
}

if ! PYTHON="$(resolve_python)"; then
    echo "Error: no Python interpreter found on PATH" >&2
    exit 1
fi

if [[ "${INSTALL_DEPS}" == true ]]; then
    echo "Installing PE Database dependencies into: ${PYTHON}"
    "${PYTHON}" -m pip install -r "${SERVICE_DIR}/requirements.txt"
    # pe-db only needs constants/sequence_utils; skip torch/tensorflow/viennarna.
    "${PYTHON}" -m pip install -e "${REPO_ROOT}/packages/pe-common" --no-deps
    echo "Dependencies installed."
fi

if ! (cd "${SERVICE_DIR}" && "${PYTHON}" -c "import uvicorn, app.main"); then
    echo "Error: missing Python dependencies (uvicorn and/or app.main)." >&2
    echo "Active Python: ${PYTHON}" >&2
    echo "Run: $(basename "$0") --install" >&2
    exit 1
fi

cd "${SERVICE_DIR}"

UVICORN_ARGS=(
    app.main:app
    --host "${HOST}"
    --port "${PORT}"
    --timeout-graceful-shutdown 5
)
if [[ "${RELOAD}" == true ]]; then
    UVICORN_ARGS+=(--reload)
fi

echo "======================================"
echo "PE Database Service"
echo "======================================"
echo "Python:      ${PYTHON}"
echo "Repo root:   ${REPO_ROOT}"
echo "Data root:   ${DATA_ROOT}"
echo "Service dir: ${SERVICE_DIR}"
echo "URL:         http://localhost:${PORT}"
echo "Docs:        http://localhost:${PORT}/docs"
echo "Reload:      ${RELOAD}"
echo "Force export:${FORCE_REEXPORT}"
echo "======================================"

exec "${PYTHON}" -m uvicorn "${UVICORN_ARGS[@]}"

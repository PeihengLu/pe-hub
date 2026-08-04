#!/usr/bin/env bash
#
# Start PE Database, PE Ensemble API, and PE Hub (unified frontend) together.
#
# Usage:
#   ./start-all.sh              # dev servers with reload (default)
#   ./start-all.sh --install    # install deps for all three, then start
#   ./start-all.sh --no-reload  # backends without uvicorn auto-reload
#   ./start-all.sh --pe-db-port 8080 --ensemble-port 8081 --frontend-port 3000
#   ./start-all.sh --force-reexport  # re-export raw data before starting PE-DB
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PE_DB_DIR="${REPO_ROOT}/services/pe-db"
PE_ENSEMBLE_DIR="${REPO_ROOT}/services/pe-ensemble"
FRONTEND_DIR="${REPO_ROOT}/pe-hub"

INSTALL_DEPS=false
RELOAD=true
FORCE_REEXPORT=false
PE_DB_HOST="${PE_DB_HOST:-0.0.0.0}"
PE_ENSEMBLE_HOST="${PE_ENSEMBLE_HOST:-0.0.0.0}"
# Set via CLI, environment, .env, or defaults (in that precedence).
PE_DB_PORT_CLI=""
PE_ENSEMBLE_PORT_CLI=""
FRONTEND_PORT_CLI=""

PIDS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start PE Database, PE Ensemble API, and PE Hub (Vite frontend).

Options:
  --install                 Install Python and npm dependencies before starting
  --no-reload               Disable uvicorn auto-reload on both backends
  --force-reexport          Re-export raw study files on PE-DB startup (also re-standardizes)
  --pe-db-port PORT         PE Database listen port (default: \$PE_DB_PORT or 8000)
  --ensemble-port PORT      PE Ensemble API listen port (default: \$PE_ENSEMBLE_PORT or 8001)
  --frontend-port PORT      PE Hub dev server port (default: \$FRONTEND_PORT or 5173)
  -h, --help                Show this help message

Environment:
  Loads \${REPO_ROOT}/.env when present.
  PE_DB_URL defaults to http://localhost:<pe-db-port>
  VITE_ENSEMBLE_API_URL defaults to http://localhost:<ensemble-port>
  --force-reexport sets PE_DB_FORCE_EXPORT / PE_DB_FORCE_STANDARDIZE for PE-DB startup.

Press Ctrl+C to stop all services.

Examples:
  ./start-all.sh
  ./start-all.sh --install
  ./start-all.sh --pe-db-port 8080 --ensemble-port 8081 --frontend-port 3000
  ./start-all.sh --force-reexport
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
        --pe-db-port)
            PE_DB_PORT_CLI="${2:?--pe-db-port requires a value}"
            shift 2
            ;;
        --ensemble-port)
            PE_ENSEMBLE_PORT_CLI="${2:?--ensemble-port requires a value}"
            shift 2
            ;;
        --frontend-port)
            FRONTEND_PORT_CLI="${2:?--frontend-port requires a value}"
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

PE_DB_PORT="${PE_DB_PORT_CLI:-${PE_DB_PORT:-8000}}"
PE_ENSEMBLE_PORT="${PE_ENSEMBLE_PORT_CLI:-${PE_ENSEMBLE_PORT:-8001}}"
FRONTEND_PORT="${FRONTEND_PORT_CLI:-${FRONTEND_PORT:-5173}}"

export PE_PROJECT_ROOT="${PE_PROJECT_ROOT:-${REPO_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-${PE_DATA_ROOT:-${REPO_ROOT}/datasets}}"

if [[ "${FORCE_REEXPORT}" == true ]]; then
    export PE_DB_FORCE_EXPORT=1
    export PE_DB_FORCE_STANDARDIZE=1
fi

export PE_DB_URL="${PE_DB_URL:-http://localhost:${PE_DB_PORT}}"
export VITE_PE_DB_URL="${VITE_PE_DB_URL:-http://localhost:${PE_DB_PORT}}"
export VITE_ENSEMBLE_API_URL="${VITE_ENSEMBLE_API_URL:-${VITE_API_URL:-http://localhost:${PE_ENSEMBLE_PORT}}}"

# CLI port flags override baked-in URLs from .env when ports were not customized there.
if [[ -n "${PE_DB_PORT_CLI}" ]]; then
    export PE_DB_URL="http://localhost:${PE_DB_PORT}"
    export VITE_PE_DB_URL="http://localhost:${PE_DB_PORT}"
fi
if [[ -n "${PE_ENSEMBLE_PORT_CLI}" ]]; then
    export VITE_ENSEMBLE_API_URL="http://localhost:${PE_ENSEMBLE_PORT}"
fi

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

MIN_NODE_MAJOR=18

node_major_version() {
    local node_bin="$1"
    "$node_bin" -p "Number(process.versions.node.split('.')[0])" 2>/dev/null
}

node_version_ok() {
    local node_bin="$1"
    local major
    major="$(node_major_version "${node_bin}")" || return 1
    (( major >= MIN_NODE_MAJOR ))
}

resolve_node() {
    local -a candidates=()
    if [[ -n "${NODE_BIN:-}" && -x "${NODE_BIN}" ]]; then
        candidates+=("${NODE_BIN}")
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/node" ]]; then
        candidates+=("${CONDA_PREFIX}/bin/node")
    fi
    if [[ -x "${HOME}/miniconda3/bin/node" ]]; then
        candidates+=("${HOME}/miniconda3/bin/node")
    fi
    if [[ -x "${HOME}/anaconda3/bin/node" ]]; then
        candidates+=("${HOME}/anaconda3/bin/node")
    fi
    if command -v node >/dev/null 2>&1; then
        candidates+=("$(command -v node)")
    fi

    local candidate seen="" deduped=()
    for candidate in "${candidates[@]}"; do
        if [[ ":${seen}:" != *":${candidate}:"* ]]; then
            deduped+=("${candidate}")
            seen="${seen}:${candidate}"
        fi
    done

    local node_bin
    for node_bin in "${deduped[@]}"; do
        if node_version_ok "${node_bin}"; then
            echo "${node_bin}"
            return 0
        fi
    done
    return 1
}

resolve_npm() {
    local node_bin npm_bin node_dir
    if node_bin="$(resolve_node)"; then
        node_dir="$(dirname "${node_bin}")"
        if [[ -x "${node_dir}/npm" ]]; then
            echo "${node_dir}/npm"
            return 0
        fi
    fi
    if command -v npm >/dev/null 2>&1; then
        npm_bin="$(command -v npm)"
        if node_version_ok "$(dirname "${npm_bin}")/node" 2>/dev/null || node_version_ok "$(command -v node 2>/dev/null)"; then
            echo "${npm_bin}"
            return 0
        fi
    fi
    return 1
}

wait_for_health() {
    local url="$1"
    local label="$2"
    local timeout="${3:-90}"
    local elapsed=0

    echo "Waiting for ${label} at ${url} ..."
    while (( elapsed < timeout )); do
        if curl -sf "${url}" >/dev/null 2>&1; then
            echo "${label} is up."
            return 0
        fi
        sleep 1
        (( elapsed += 1 )) || true
    done

    echo "Error: timed out waiting for ${label} (${url})" >&2
    return 1
}

cleanup() {
    local pid
    echo ""
    echo "Stopping services ..."
    # Guard empty PIDS under `set -u` (bash 3.2 / macOS default).
    if ((${#PIDS[@]})); then
        for pid in "${PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" 2>/dev/null || true
            fi
        done
    fi
    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

start_pe_db() {
    local -a uvicorn_args=(
        app.main:app
        --host "${PE_DB_HOST}"
        --port "${PE_DB_PORT}"
        --timeout-graceful-shutdown 5
    )
    if [[ "${RELOAD}" == true ]]; then
        uvicorn_args+=(
            --reload
            --reload-dir "${PE_DB_DIR}/app"
            --reload-dir "${REPO_ROOT}/packages/pe-common/pe_common"
        )
    fi

    (
        cd "${PE_DB_DIR}"
        exec "${PYTHON}" -m uvicorn "${uvicorn_args[@]}"
    ) &
    PIDS+=("$!")
}

start_pe_ensemble() {
    local -a uvicorn_args=(
        app.main:app
        --host "${PE_ENSEMBLE_HOST}"
        --port "${PE_ENSEMBLE_PORT}"
        --timeout-graceful-shutdown 5
    )
    if [[ "${RELOAD}" == true ]]; then
        uvicorn_args+=(
            --reload
            --reload-dir "${PE_ENSEMBLE_DIR}/app"
            --reload-dir "${REPO_ROOT}/packages/pe-common/pe_common"
        )
    fi

    (
        cd "${PE_ENSEMBLE_DIR}"
        export PE_DB_URL
        exec "${PYTHON}" -m uvicorn "${uvicorn_args[@]}"
    ) &
    PIDS+=("$!")
}

start_frontend() {
    local NODE_BIN
    if ! NODE_BIN="$(resolve_node)"; then
        echo "Error: Node.js ${MIN_NODE_MAJOR}+ is required for the PE Hub frontend (Vite 5)." >&2
        echo "Install Node 20+ (e.g. conda install nodejs=20) or set NODE_BIN to a modern node binary." >&2
        if command -v node >/dev/null 2>&1; then
            echo "Found $(command -v node) ($(node --version 2>/dev/null || echo unknown))" >&2
        fi
        exit 1
    fi
    if ! NPM="$(resolve_npm)"; then
        echo "Error: npm is not installed next to ${NODE_BIN}" >&2
        exit 1
    fi

    (
        cd "${FRONTEND_DIR}"
        export PATH="$(dirname "${NODE_BIN}"):${PATH}"
        export VITE_PE_DB_URL
        export VITE_ENSEMBLE_API_URL
        exec "${NPM}" run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
    ) &
    PIDS+=("$!")
}

if [[ "${INSTALL_DEPS}" == true ]]; then
    echo "Installing PE Database dependencies ..."
    "${PYTHON}" -m pip install -r "${PE_DB_DIR}/requirements.txt"
    "${PYTHON}" -m pip install -e "${REPO_ROOT}/packages/pe-common" --no-deps

    echo "Installing PE Ensemble dependencies ..."
    "${PYTHON}" -m pip install -e "${REPO_ROOT}/packages/pe-common"
    "${PYTHON}" -m pip install -e "${PE_ENSEMBLE_DIR}"

    echo "Installing frontend dependencies ..."
    (cd "${FRONTEND_DIR}" && npm install)
    echo "Dependencies installed."
fi

if ! (cd "${PE_DB_DIR}" && "${PYTHON}" -c "import uvicorn, app.main"); then
    echo "Error: PE Database dependencies missing. Run: $(basename "$0") --install" >&2
    exit 1
fi

if ! (cd "${PE_ENSEMBLE_DIR}" && "${PYTHON}" -c "import uvicorn, prettytable"); then
    echo "Error: PE Ensemble dependencies missing (need uvicorn and prettytable for PRIDICT2)." >&2
    echo "Run: $(basename "$0") --install" >&2
    echo "Active Python: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
    if ! resolve_npm >/dev/null; then
        echo "Error: frontend node_modules missing and npm is not installed." >&2
        echo "Run: $(basename "$0") --install" >&2
        exit 1
    fi
    echo "Frontend node_modules not found; run: $(basename "$0") --install" >&2
    exit 1
fi

echo "======================================"
echo "PE Platform (all services)"
echo "======================================"
echo "Python:         ${PYTHON}"
if NODE_BIN="$(resolve_node)"; then
    echo "Node.js:        ${NODE_BIN} ($("${NODE_BIN}" --version))"
else
    echo "Node.js:        not found (need ${MIN_NODE_MAJOR}+)"
fi
echo "Data root:      ${DATA_ROOT}"
echo "PE Database:    http://localhost:${PE_DB_PORT}  (docs: /docs)"
echo "PE Ensemble:    http://localhost:${PE_ENSEMBLE_PORT}  (docs: /docs)"
echo "PE Hub:         http://localhost:${FRONTEND_PORT}"
echo "PE_DB_URL:      ${PE_DB_URL}"
echo "VITE_PE_DB_URL: ${VITE_PE_DB_URL}"
echo "VITE_ENSEMBLE:  ${VITE_ENSEMBLE_API_URL}"
echo "Reload:         ${RELOAD}"
echo "Force export:   ${FORCE_REEXPORT}"
echo "======================================"
echo ""

start_pe_db
wait_for_health "http://127.0.0.1:${PE_DB_PORT}/health" "PE Database"

start_pe_ensemble
wait_for_health "http://127.0.0.1:${PE_ENSEMBLE_PORT}/health" "PE Ensemble API"

start_frontend

echo ""
echo "All services started. Press Ctrl+C to stop."
echo ""

wait

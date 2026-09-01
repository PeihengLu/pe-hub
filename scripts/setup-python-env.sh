#!/usr/bin/env bash
# Create or refresh the supported PE-Hub Python environment (3.11).
#
# Usage (from repo root):
#   ./scripts/setup-python-env.sh --install    # create env + run install-clis.sh
#   ./scripts/setup-python-env.sh --venv       # repo-local venv/ (needs python3.11)
#   ./scripts/setup-python-env.sh --name pedb  # custom conda env name
#
# After creation (without --install):
#   conda activate pe-hub    # or: source venv/bin/activate
#   ./scripts/install-clis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/python-env.sh
source "${REPO_ROOT}/scripts/python-env.sh"

USE_VENV=false
ENV_NAME="pe-hub"
RUN_INSTALL=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create the PE-Hub Python ${PE_HUB_PYTHON_VERSION} environment.

Options:
  --install       After creating the env, run ./scripts/install-clis.sh
  --venv          Create ${REPO_ROOT}/venv with python3.11 (no conda)
  --name NAME     Conda environment name (default: pe-hub)
  -h, --help      Show this help

Examples:
  ./scripts/setup-python-env.sh --install
  ./scripts/setup-python-env.sh --name pedb --install
  ./scripts/setup-python-env.sh --venv --install
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            RUN_INSTALL=true
            shift
            ;;
        --venv)
            USE_VENV=true
            shift
            ;;
        --name)
            ENV_NAME="${2:?--name requires a value}"
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

if [[ "${USE_VENV}" == true ]]; then
    if ! command -v "python${PE_HUB_PYTHON_VERSION}" >/dev/null 2>&1; then
        echo "Error: python${PE_HUB_PYTHON_VERSION} not found on PATH." >&2
        echo "Install Python ${PE_HUB_PYTHON_VERSION} or use conda instead:" >&2
        echo "  ./scripts/setup-python-env.sh" >&2
        exit 1
    fi
    echo "Creating venv at ${REPO_ROOT}/venv (python${PE_HUB_PYTHON_VERSION}) ..."
    "python${PE_HUB_PYTHON_VERSION}" -m venv "${REPO_ROOT}/venv"
  # shellcheck disable=SC1091
    source "${REPO_ROOT}/venv/bin/activate"
    python -m pip install --upgrade pip wheel
    if [[ "${RUN_INSTALL}" == true ]]; then
        bash "${REPO_ROOT}/scripts/install-clis.sh"
    else
        echo ""
        echo "Activate with:  source ${REPO_ROOT}/venv/bin/activate"
        echo "Then install:   ./scripts/install-clis.sh"
    fi
    exit 0
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found." >&2
    echo "Install Miniconda/Mambaforge, or use a venv:" >&2
    echo "  ./scripts/setup-python-env.sh --venv" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_FILE="${REPO_ROOT}/environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Error: missing ${ENV_FILE}" >&2
    exit 1
fi

if [[ "${ENV_NAME}" != "pe-hub" ]]; then
    echo "Creating conda env '${ENV_NAME}' (python ${PE_HUB_PYTHON_VERSION}) ..."
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "Environment '${ENV_NAME}' already exists."
    else
        conda create -y -n "${ENV_NAME}" "python=${PE_HUB_PYTHON_VERSION}" pip
    fi
    conda activate "${ENV_NAME}"
    conda install -y -c conda-forge -c bioconda viennarna "nodejs=20"
else
    if conda env list | awk '{print $1}' | grep -qx "pe-hub"; then
        echo "Updating existing conda env 'pe-hub' from environment.yml ..."
        conda env update -f "${ENV_FILE}" --prune
    else
        echo "Creating conda env 'pe-hub' from environment.yml ..."
        conda env create -f "${ENV_FILE}"
    fi
    conda activate pe-hub
fi

python -m pip install --upgrade pip wheel

if [[ "${RUN_INSTALL}" == true ]]; then
    bash "${REPO_ROOT}/scripts/install-clis.sh"
else
    echo ""
    echo "Environment ready. Activate with:"
    echo "  conda activate ${ENV_NAME}"
    echo "Then install project packages:"
    echo "  ./scripts/install-clis.sh"
fi

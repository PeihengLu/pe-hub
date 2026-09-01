#!/usr/bin/env bash
# One-time / occasional setup on Oxford ARC (run on an *interactive* HTC node).
#
#   ssh …@htc-login.arc.ox.ac.uk
#   srun -p interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
#   export PE_HUB_ROOT=$DATA/pe-hub   # or your checkout
#   bash $PE_HUB_ROOT/scripts/cluster/oxford-arc/setup_interactive.sh
#
# Loads Anaconda via ARC modules, creates a $DATA env prefix, installs peen.
# Does not submit training.

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi

: "${PE_HUB_ROOT:?Set PE_HUB_ROOT to the pe-hub checkout under \$DATA}"
ARC_MODULES="${ARC_MODULES:-Anaconda3}"
CONDA_ENV="${CONDA_ENV:-${DATA:+$DATA/envs/pe-hub}}"
: "${CONDA_ENV:?Set CONDA_ENV to a prefix under \$DATA (e.g. \$DATA/envs/pe-hub)}"

echo "PE_HUB_ROOT=${PE_HUB_ROOT}"
echo "ARC_MODULES=${ARC_MODULES}"
echo "CONDA_ENV=${CONDA_ENV}"

if [[ ! -d "${PE_HUB_ROOT}/.git" && ! -f "${PE_HUB_ROOT}/README.md" ]]; then
    echo "Error: PE_HUB_ROOT does not look like pe-hub: ${PE_HUB_ROOT}" >&2
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi
if ! command -v module >/dev/null 2>&1; then
    echo "Error: environment modules not available on this node." >&2
    exit 1
fi

echo "Loading modules: ${ARC_MODULES}"
# shellcheck disable=SC2086
module load ${ARC_MODULES}

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not on PATH after module load." >&2
    echo "  Try: module spider Anaconda   # or Mamba" >&2
    echo "  Then set ARC_MODULES in env.sh to the exact module name." >&2
    exit 1
fi

# Prefer conda.sh when present; ARC also supports: source activate $PREFIX
if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

ENV_FILE="${PE_HUB_ROOT}/environment.yml"
PYTHON_PIN="python=3.11.*"

_verify_conda_python_311() {
    local ver minor
    ver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
    minor="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${minor}" != "3.11" ]]; then
        echo "Error: ${CONDA_ENV} has Python ${ver}; PE-Hub requires 3.11.x." >&2
        echo "Conda likely upgraded Python while installing bioconda packages." >&2
        echo "Remove the env and re-run this script:" >&2
        echo "  conda deactivate" >&2
        echo "  rm -rf \"${CONDA_ENV}\"" >&2
        echo "  bash \"${ARC_DIR}/setup_interactive.sh\"" >&2
        exit 1
    fi
}

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "Creating conda env prefix ${CONDA_ENV} (${PYTHON_PIN})…"
    mkdir -p "$(dirname "${CONDA_ENV}")"
    if [[ -f "${ENV_FILE}" ]]; then
        conda env create --prefix "${CONDA_ENV}" -f "${ENV_FILE}"
    else
        conda create -y --prefix "${CONDA_ENV}" "${PYTHON_PIN}" pip
    fi
fi

# Activate prefix (works with named envs too).
conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"
_verify_conda_python_311

echo "Installing ViennaRNA (bioconda) if missing…"
if ! python -c "import RNA" 2>/dev/null; then
    # Pin Python — bare `conda install viennarna` can pull 3.13 on newer solvers.
    conda install -y -c bioconda -c conda-forge "${PYTHON_PIN}" viennarna
    _verify_conda_python_311
fi

echo "Installing PyTorch (CUDA wheel). Adjust cu121/cu124 if needed for the node driver…"
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    pip install "torch>=2.0,<2.9" --index-url https://download.pytorch.org/whl/cu124

cd "${PE_HUB_ROOT}"
./scripts/install-clis.sh

if ! command -v dvc >/dev/null 2>&1; then
    echo "Installing dvc (ARC local remote does not need SSH, extra is harmless)…"
    pip install 'dvc[ssh]'
fi

echo "Seeding / preparing datasets if needed (may take a while)…"
if [[ ! -d "${PE_HUB_ROOT}/datasets/standardized" ]]; then
    pedb init || true
fi

echo ""
echo "Smoke-check:"
peen devices
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

echo ""
echo "Ready. Submit from htc-login (with env.sh configured):"
echo "  ${ARC_DIR}/submit.sh 01_tune_base_library1.sh"
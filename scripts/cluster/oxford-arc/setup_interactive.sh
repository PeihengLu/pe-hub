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

if [[ ! -d "${CONDA_ENV}" ]]; then
    echo "Creating conda env prefix ${CONDA_ENV} (python 3.11)…"
    mkdir -p "$(dirname "${CONDA_ENV}")"
    conda create -y --prefix "${CONDA_ENV}" python=3.11 pip
fi

# Activate prefix (works with named envs too).
conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"

echo "Installing ViennaRNA (bioconda) if missing…"
python -c "import RNA" 2>/dev/null || conda install -y -c bioconda -c conda-forge viennarna

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
#!/usr/bin/env bash
# One-time / occasional setup on Oxford ARC (run on an *interactive* HTC node).
#
#   ssh …@htc-login.arc.ox.ac.uk
#   srun -p interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
#   export PE_HUB_ROOT=$DATA/pe-hub   # or your checkout
#   bash $PE_HUB_ROOT/scripts/cluster/oxford-arc/setup_interactive.sh
#
# Does not submit training; only prepares conda + CLIs and smoke-checks peen.

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi

: "${PE_HUB_ROOT:?Set PE_HUB_ROOT to the pe-hub checkout under \$DATA}"
CONDA_ROOT="${CONDA_ROOT:-$DATA/conda}"
CONDA_ENV="${CONDA_ENV:-pe-hub}"

echo "PE_HUB_ROOT=${PE_HUB_ROOT}"
echo "CONDA_ROOT=${CONDA_ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"

if [[ ! -d "${PE_HUB_ROOT}/.git" && ! -f "${PE_HUB_ROOT}/README.md" ]]; then
    echo "Error: PE_HUB_ROOT does not look like pe-hub: ${PE_HUB_ROOT}" >&2
    exit 1
fi

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "Miniconda not found at ${CONDA_ROOT}."
    echo "Install under \$DATA (HOME is 15 GiB):"
    echo "  cd \$DATA && curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_ROOT}"
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "Creating conda env ${CONDA_ENV} (python 3.11)…"
    conda create -y -n "${CONDA_ENV}" python=3.11 pip
fi
conda activate "${CONDA_ENV}"

echo "Installing ViennaRNA (bioconda) if missing…"
python -c "import RNA" 2>/dev/null || conda install -y -c bioconda -c conda-forge viennarna

echo "Installing PyTorch (CUDA wheel). Adjust cu121/cu124 if needed for the node driver…"
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    pip install "torch>=2.0,<2.9" --index-url https://download.pytorch.org/whl/cu124

cd "${PE_HUB_ROOT}"
./scripts/install-clis.sh

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
echo "  ${ARC_DIR}/submit.sh 03_train_base_library1.sh"

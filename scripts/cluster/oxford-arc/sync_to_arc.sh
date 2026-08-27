#!/usr/bin/env bash
# Push pe-hub source + raw/standardized data to Oxford ARC.
# Run on your laptop (VPN → htc-login, or gateway off-net).
#
# Skips training/eval artifacts and regenerated dataset caches so you do not
# overwrite cluster outputs or upload large convertible trees.
#
# Required (or set in env.sh):
#   ARC_REMOTE   e.g. wolf6973@htc-login.arc.ox.ac.uk:/data/.../pe-hub
#
# Examples:
#   ./scripts/cluster/oxford-arc/sync_to_arc.sh
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/sync_to_arc.sh
#   # Also omit standardized (rebuild on ARC via pedb init / setup_interactive):
#   SKIP_STANDARDIZED=1 ./scripts/cluster/oxford-arc/sync_to_arc.sh

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "${ARC_DIR}/../../.." && pwd)}"

if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi

: "${ARC_REMOTE:?Set ARC_REMOTE=user@host:/data/.../pe-hub}"

RSYNC_FLAGS=(-avz --progress)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RSYNC_FLAGS+=(--dry-run)
fi

# Always exclude bulky / regenerable / cluster-owned artifacts.
EXCLUDES=(
    # VCS / caches / IDE
    --exclude='.git/objects/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.egg-info/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
    --exclude='.ruff_cache/'
    --exclude='.DS_Store'
    --exclude='.idea/'
    --exclude='.vscode/'
    --exclude='.cursor/'
    --exclude='node_modules/'

    # Converted / generated dataset caches (rebuild on ARC if needed)
    --exclude='datasets/exported/'
    --exclude='datasets/formatted/'
    --exclude='datasets/catalog/'

    # Model training / eval / ensemble artifacts
    --exclude='services/pe-ensemble/weights/'
    --exclude='services/pe-ensemble/tuning_studies/'
    --exclude='services/pe-ensemble/config/training_presets_local/'
    --exclude='services/pe-ensemble/jobs/'
    --exclude='services/pe-ensemble/eval_jobs/'
    --exclude='services/pe-ensemble/ensemble_jobs/'
    --exclude='services/pe-ensemble/validation_jobs/'
    --exclude='services/pe-ensemble/checkpoints/'
    --exclude='artifacts/'
    --exclude='checkpoints/'

    # Experiment pipeline state (cluster-owned after submit)
    --exclude='scripts/experiments/pridict2-reproduction/state/'

    # Slurm logs at checkout root
    --exclude='slurm-*.out'
    --exclude='slurm-*.err'
)

if [[ "${SKIP_STANDARDIZED:-0}" == "1" ]]; then
    EXCLUDES+=(--exclude='datasets/standardized/')
fi

# Vendor model sources must be checked out locally before rsync (ARC often
# cannot clone git@github.com submodules without extra SSH setup).
vendor_empty=0
for name in deepprime oped pridict2 optiprime; do
    if [[ ! -d "${LOCAL_ROOT}/vendor/models/${name}" ]] || \
       [[ -z "$(find "${LOCAL_ROOT}/vendor/models/${name}" -mindepth 1 -maxdepth 1 ! -name '.git' 2>/dev/null | head -1)" ]]; then
        echo "Error: vendor/models/${name} is empty." >&2
        vendor_empty=1
    fi
done
if [[ "${vendor_empty}" -ne 0 ]]; then
    echo "Init submodules locally first, then re-run sync:" >&2
    echo "  git submodule update --init --recursive" >&2
    echo "  # or: ./scripts/install-clis.sh" >&2
    exit 1
fi

echo "From: ${LOCAL_ROOT}/"
echo "To:   ${ARC_REMOTE}/"
echo "Excludes: exported/formatted/catalog, weights, jobs, eval/ensemble/validation,"
echo "          tuning_studies, presets_local, artifacts, experiment state"
if [[ "${SKIP_STANDARDIZED:-0}" == "1" ]]; then
    echo "          + datasets/standardized (SKIP_STANDARDIZED=1)"
else
    echo "Keeps: datasets/raw + datasets/standardized + vendor/models (omit standardized with SKIP_STANDARDIZED=1)"
fi
echo ""

rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" \
    "${LOCAL_ROOT}/" \
    "${ARC_REMOTE}/"

echo ""
echo "Done. On ARC (interactive GPU node) if peen env is not ready:"
echo "  bash \$PE_HUB_ROOT/scripts/cluster/oxford-arc/setup_interactive.sh"

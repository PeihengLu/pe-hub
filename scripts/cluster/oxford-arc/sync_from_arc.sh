#!/usr/bin/env bash
# Pull cluster training outputs to your local pe-hub checkout.
# Run on your laptop (not on ARC login nodes).
#
# Required:
#   ARC_REMOTE   e.g. you@gateway.arc.ox.ac.uk:/data/<project>/<user>/pe-hub
#   LOCAL_ROOT   local pe-hub path (default: repo containing this script)
#
# Examples:
#   ARC_REMOTE=you@gateway.arc.ox.ac.uk:/data/proj/you/pe-hub \
#     ./scripts/cluster/oxford-arc/sync_from_arc.sh
#
#   # Also pull Optuna DBs (optional resume on laptop):
#   SYNC_STUDIES=1 ARC_REMOTE=... ./scripts/cluster/oxford-arc/sync_from_arc.sh
#
# Pulls (gitignored / local):
#   - trained weight dirs (model__scope__date__id) + local_registry.json
#   - training_presets_local/*.yaml  (HPO hits — do not commit routinely)
#   - pridict2-reproduction state/
# Does not require pushing presets to GitHub.

set -euo pipefail

# ARC_DIR is the directory of the local script on your laptop
ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "${ARC_DIR}/../../.." && pwd)}"

if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi

: "${ARC_REMOTE:?Set ARC_REMOTE=user@host:/data/.../pe-hub}"

WEIGHTS_REL="services/pe-ensemble/weights"
LOCAL_PRESETS_REL="services/pe-ensemble/config/training_presets_local"
STATE_REL="scripts/experiments/pridict2-reproduction/state"
STUDIES_REL="services/pe-ensemble/tuning_studies"

echo "From: ${ARC_REMOTE}"
echo "To:   ${LOCAL_ROOT}"
echo ""

mkdir -p "${LOCAL_ROOT}/${WEIGHTS_REL}"
rsync -avz --progress \
    --include='local_registry.json' \
    --include='*/' \
    --include='*__*__*__*/' \
    --include='*__*__*__*/**' \
    --exclude='*' \
    "${ARC_REMOTE}/${WEIGHTS_REL}/" \
    "${LOCAL_ROOT}/${WEIGHTS_REL}/"

mkdir -p "${LOCAL_ROOT}/${LOCAL_PRESETS_REL}"
rsync -avz --progress \
    --exclude='.gitignore' \
    "${ARC_REMOTE}/${LOCAL_PRESETS_REL}/" \
    "${LOCAL_ROOT}/${LOCAL_PRESETS_REL}/" || echo "(no local presets on remote yet — ok)"

if rsync -avz --progress \
    "${ARC_REMOTE}/${STATE_REL}/" \
    "${LOCAL_ROOT}/${STATE_REL}/" 2>/dev/null; then
    :
else
    echo "(no state/ on remote yet — ok)"
fi

if [[ "${SYNC_STUDIES:-0}" == "1" ]]; then
    mkdir -p "${LOCAL_ROOT}/${STUDIES_REL}"
    rsync -avz --progress \
        "${ARC_REMOTE}/${STUDIES_REL}/" \
        "${LOCAL_ROOT}/${STUDIES_REL}/"
fi

echo ""
echo "Done. Local HPO presets and trained weights are gitignored."
echo "  peen weights --model pridict2"
echo "  ./scripts/hyperparameter/check_tuning_status.sh pridict2"
echo "Only promote curated baselines into config/training_presets/ when publishing."

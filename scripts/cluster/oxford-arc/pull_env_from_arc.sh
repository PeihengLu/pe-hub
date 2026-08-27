#!/usr/bin/env bash
# Pull gitignored env.sh from your ARC pe-hub checkout to this laptop.
# Run locally (VPN → htc-login, or set ARC_HOST=gateway.arc.ox.ac.uk off-net).
#
# Usage:
#   ./scripts/cluster/oxford-arc/pull_env_from_arc.sh
#   ./scripts/cluster/oxford-arc/pull_env_from_arc.sh wolf6973
#   ARC_USER=wolf6973 ./scripts/cluster/oxford-arc/pull_env_from_arc.sh
#
# Optional env:
#   ARC_HOST     default htc-login.arc.ox.ac.uk
#   ARC_PROJECT  default coml-deepcmb  → remote path /data/$ARC_PROJECT/$USER/pe-hub
#   ARC_REMOTE   if set, used as-is (skips host/project/user construction)
#   DRY_RUN=1    rsync --dry-run

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="${ARC_DIR}/env.sh"

ARC_HOST="${ARC_HOST:-htc-login.arc.ox.ac.uk}"
ARC_PROJECT="${ARC_PROJECT:-coml-deepcmb}"

if [[ -n "${1:-}" ]]; then
    ARC_USER="$1"
elif [[ -z "${ARC_USER:-}" ]]; then
    read -r -p "ARC username: " ARC_USER
fi
: "${ARC_USER:?ARC username required}"

if [[ -z "${ARC_REMOTE:-}" ]]; then
    ARC_REMOTE="${ARC_USER}@${ARC_HOST}:/data/${ARC_PROJECT}/${ARC_USER}/pe-hub"
fi

REMOTE_ENV="${ARC_REMOTE}/scripts/cluster/oxford-arc/env.sh"

RSYNC_FLAGS=(-avz --progress)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RSYNC_FLAGS+=(--dry-run)
fi

echo "From: ${REMOTE_ENV}"
echo "To:   ${LOCAL_ENV}"
echo ""

rsync "${RSYNC_FLAGS[@]}" "${REMOTE_ENV}" "${LOCAL_ENV}"

echo ""
echo "Done. Local env.sh updated (gitignored)."
echo "  grep ARC_PARTITION ${LOCAL_ENV}"

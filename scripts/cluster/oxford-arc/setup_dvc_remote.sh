#!/usr/bin/env bash
# Write .dvc/config.local so the `arc` remote works on this machine.
# Does not commit anything (config.local is gitignored).
#
# Default store: /data/<ARC_PROJECT>/<ARC_USER>/dvc-store
# (same layout as pe-hub under $DATA). The project is shared; each user has
# their own dvc-store. Override with DVC_STORE if needed.
#
# Usage:
#   ./scripts/cluster/oxford-arc/setup_dvc_remote.sh
#   ARC_USER=wolf6973 ./scripts/cluster/oxford-arc/setup_dvc_remote.sh
#   ARC_PROJECT=coml-deepcmb ARC_USER=wolf6973 ./scripts/cluster/oxford-arc/setup_dvc_remote.sh

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ARC_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi

ARC_PROJECT="${ARC_PROJECT:-coml-deepcmb}"
ARC_HOST="${ARC_HOST:-htc-login.arc.ox.ac.uk}"

if [[ -z "${ARC_USER:-}" ]]; then
    ARC_USER="${USER:?Set ARC_USER to your ARC username}"
fi

# Prefer $DATA/dvc-store when already on ARC ($DATA == /data/<project>/<user>).
if [[ -n "${DVC_STORE:-}" ]]; then
    STORE="${DVC_STORE}"
elif [[ -n "${DATA:-}" && "${DATA}" == /data/* ]]; then
    STORE="${DATA}/dvc-store"
else
    STORE="/data/${ARC_PROJECT}/${ARC_USER}/dvc-store"
fi

if ! command -v dvc >/dev/null 2>&1; then
    echo "Error: dvc not on PATH. conda activate pedb && pip install 'dvc[ssh]'" >&2
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/.dvc/config" ]]; then
    echo "Error: ${REPO_ROOT} is not a DVC project (missing .dvc/config)." >&2
    exit 1
fi

on_arc=0
if [[ -n "${DATA:-}" && "${DATA}" == /data/* ]] || [[ -d /data/${ARC_PROJECT}/${ARC_USER} ]]; then
    on_arc=1
fi

if [[ "${on_arc}" -eq 1 ]]; then
    echo "ARC filesystem remote:"
    echo "  ${STORE}"
    dvc remote modify --local arc url "${STORE}"
    mkdir -p "${STORE}"
else
    URL="ssh://${ARC_USER}@${ARC_HOST}${STORE}"
    echo "Laptop SSH remote (VPN or ProxyJump to ${ARC_HOST}):"
    echo "  ${URL}"
    dvc remote modify --local arc url "${URL}"
    dvc remote modify --local arc user "${ARC_USER}"
fi

echo ""
echo "Wrote ${REPO_ROOT}/.dvc/config.local"
if [[ "${on_arc}" -eq 0 ]]; then
    echo "On ARC, create the store once:  mkdir -p ${STORE}"
fi
echo "Then:  dvc push    # from the machine that has the data"
echo "       dvc pull    # on the other machine"

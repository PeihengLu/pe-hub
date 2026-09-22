#!/usr/bin/env bash
# Push DVC-tracked folders, gitignored env.sh, and vendor-eval results to ARC.
# Run locally (VPN → htc-login, or ProxyJump via gateway off-net).
#
# Usage:
#   ./scripts/cluster/oxford-arc/push_to_arc.sh
#   ./scripts/cluster/oxford-arc/push_to_arc.sh YOUR_ARC_USER
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/push_to_arc.sh
#   ONLY=results,slurm_output ./scripts/cluster/oxford-arc/push_to_arc.sh

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_rsync_arc.sh
source "${ARC_DIR}/_rsync_arc.sh"
arc_rsync_main push "$@"

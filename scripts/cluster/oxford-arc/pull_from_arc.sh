#!/usr/bin/env bash
# Pull DVC-tracked folders, gitignored env.sh, and vendor-eval results from ARC.
# Run locally (VPN → htc-login, or ProxyJump via gateway off-net).
#
# Usage:
#   ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   SKIP=datasets/reference ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ONLY=env ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
#   ONLY=vendor-eval ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
#   EXTRA=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_rsync_arc.sh
source "${ARC_DIR}/_rsync_arc.sh"
arc_rsync_main pull "$@"

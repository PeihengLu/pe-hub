#!/usr/bin/env bash
# Pull DVC-tracked folders and gitignored env.sh from ARC onto this laptop.
# Run locally (VPN → htc-login, or ProxyJump via gateway off-net).
#
# Usage:
#   ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ./scripts/cluster/oxford-arc/pull_from_arc.sh wolf6973
#   DRY_RUN=1 ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   SKIP=datasets/reference ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ONLY=env ./scripts/cluster/oxford-arc/pull_from_arc.sh
#   ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh wolf6973
#   EXTRA=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh wolf6973

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_rsync_arc.sh
source "${ARC_DIR}/_rsync_arc.sh"
arc_rsync_main pull "$@"

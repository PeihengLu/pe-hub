#!/usr/bin/env bash
# Pull gitignored env.sh from ARC onto this laptop.
# For DVC-tracked folders as well, use pull_from_arc.sh.

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${ONLY:-env}" exec "${ARC_DIR}/pull_from_arc.sh" "$@"

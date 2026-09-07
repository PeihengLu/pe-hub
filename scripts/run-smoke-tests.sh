#!/usr/bin/env bash
#
# Compatibility wrapper. Prefer ./scripts/run-tests.sh.
# With no groups, this still runs every suite; pass ``smoke`` for HTTP/CLI only.
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run-tests.sh" "$@"

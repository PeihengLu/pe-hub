#!/usr/bin/env bash
# Zip PRIDICT2 reproduction weight artifacts for thesis supplementary files.
#
# Prerequisites: pull the bundle from ARC first, e.g.
#   ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh wolf6973
#
# Usage:
#   ./scripts/experiments/pridict2-reproduction/pack_supplementary_weights.sh
#   OUT=/tmp/pridict2-repro-weights.zip ./.../pack_supplementary_weights.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST="${SCRIPT_DIR}/supplementary_artifacts.txt"
OUT="${OUT:-${REPO_ROOT}/txt/supplementary/pridict2-reproduction-weights.zip}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Error: missing ${MANIFEST}" >&2
    exit 1
fi

mapfile -t RELS < <(
    awk '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*$/ { next }
        { print $1 }
    ' "${MANIFEST}"
)

missing=0
for rel in "${RELS[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${rel}" ]]; then
        echo "missing: ${rel}" >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    echo "Error: pull artifacts first:" >&2
    echo "  ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh <ARC_USER>" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"

(
    cd "${REPO_ROOT}"
    zip -r -q "${OUT}" "${RELS[@]}" \
        scripts/experiments/pridict2-reproduction/README.md \
        scripts/experiments/pridict2-reproduction/supplementary_artifacts.txt
)

echo "Wrote ${OUT}"
du -h "${OUT}"

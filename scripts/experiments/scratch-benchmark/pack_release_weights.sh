#!/usr/bin/env bash
# Pack preferred scratch-benchmark weights as a GitHub Release asset.
#
# Prerequisites: preferred weight dirs present under services/pe-ensemble/weights/
# (e.g. ONLY=scratch-weights pull from ARC), then:
#
#   python3 scripts/experiments/scratch-benchmark/publish_preferred_weights.py
#   ./scripts/experiments/scratch-benchmark/pack_release_weights.sh
#
# Upload the zip to the shared experiment-weights release:
#   gh release upload experiment-weights-v1 txt/supplementary/scratch-benchmark-weights.zip
#
# Install on another machine:
#   ./scripts/experiments/scratch-benchmark/install_release_weights.sh \
#     /path/to/scratch-benchmark-weights.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STAGE="${REPO_ROOT}/txt/supplementary/scratch-benchmark-weights"
OUT="${OUT:-${REPO_ROOT}/txt/supplementary/scratch-benchmark-weights.zip}"

cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/export_distributable_weights.py"

if [[ ! -d "${STAGE}/weights" ]]; then
    echo "Error: missing staged weights at ${STAGE}/weights" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT}")"
rm -f "${OUT}"

(
    cd "${REPO_ROOT}/txt/supplementary"
    zip -r -q "${OUT}" scratch-benchmark-weights \
        -x 'scratch-benchmark-weights/**/.DS_Store' \
        -x 'scratch-benchmark-weights/**/__pycache__/*'
)

echo "Wrote ${OUT}"
du -h "${OUT}"
echo "Upload: gh release upload experiment-weights-v1 ${OUT}"

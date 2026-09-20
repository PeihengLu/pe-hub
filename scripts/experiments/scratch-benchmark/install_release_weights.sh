#!/usr/bin/env bash
# Install scratch-benchmark preferred weights from a GitHub Release zip.
#
# Usage:
#   ./scripts/experiments/scratch-benchmark/install_release_weights.sh \\
#     /path/to/scratch-benchmark-weights.zip
#
#   # Or download then install:
#   gh release download <TAG> -p scratch-benchmark-weights.zip -D /tmp
#   ./scripts/experiments/scratch-benchmark/install_release_weights.sh \\
#     /tmp/scratch-benchmark-weights.zip
#
# Copies weight dirs into services/pe-ensemble/weights/ and relabels
# local_registry via publish_preferred_weights.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ZIP="${1:-}"

if [[ -z "${ZIP}" || ! -f "${ZIP}" ]]; then
    echo "Usage: $0 /path/to/scratch-benchmark-weights.zip" >&2
    exit 2
fi

ZIP="$(cd "$(dirname "${ZIP}")" && pwd)/$(basename "${ZIP}")"
DEST="${REPO_ROOT}/services/pe-ensemble/weights"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/scratch-weights.XXXXXX")"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "Unpacking ${ZIP} → ${TMP}"
unzip -q "${ZIP}" -d "${TMP}"

WEIGHTS_SRC=""
if [[ -d "${TMP}/scratch-benchmark-weights/weights" ]]; then
    WEIGHTS_SRC="${TMP}/scratch-benchmark-weights/weights"
elif [[ -d "${TMP}/weights" ]]; then
    WEIGHTS_SRC="${TMP}/weights"
else
    echo "Error: zip has no weights/ tree (expected scratch-benchmark-weights/weights/)" >&2
    find "${TMP}" -maxdepth 3 -type d | head -40 >&2
    exit 1
fi

mkdir -p "${DEST}"
echo "Installing weight dirs → ${DEST}"
# Copy model subtrees (deepprime/, oped/, pridict2/) without clobbering vendor IDs
# that live alongside trained sets.
for model_dir in "${WEIGHTS_SRC}"/*; do
    [[ -d "${model_dir}" ]] || continue
    model="$(basename "${model_dir}")"
    mkdir -p "${DEST}/${model}"
    for entry in "${model_dir}"/*; do
        [[ -e "${entry}" ]] || continue
        name="$(basename "${entry}")"
        target="${DEST}/${model}/${name}"
        if [[ -e "${target}" ]]; then
            echo "  replace ${model}/${name}"
            rm -rf "${target}"
        else
            echo "  add ${model}/${name}"
        fi
        cp -a "${entry}" "${target}"
    done
done

# Prefer map from the zip when present (keeps publish selection in sync).
MAP_SRC="${TMP}/scratch-benchmark-weights/weights_id_map.tsv"
if [[ -f "${MAP_SRC}" ]]; then
    cp -a "${MAP_SRC}" "${SCRIPT_DIR}/weights_id_map.tsv"
fi

python3 "${SCRIPT_DIR}/publish_preferred_weights.py"
echo "Done. Example: peen weights --model oped | grep 'Scratch holdout3' | head"

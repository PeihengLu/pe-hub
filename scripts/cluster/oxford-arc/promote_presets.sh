#!/usr/bin/env bash
# Promote selected local HPO dataset entries into shipped (git-tracked) presets.
# Merges `datasets:` only — does not replace shipped `defaults:`.
#
# Routine tuning should NOT use this — keep hits in training_presets_local/.
#
# Usage:
#   ./scripts/cluster/oxford-arc/promote_presets.sh              # dry-run
#   MODEL=pridict2 ./scripts/cluster/oxford-arc/promote_presets.sh --apply
#   KEY=pridict1/library1/hek293t/pe2 MODEL=pridict2 ... --apply

set -euo pipefail

ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ARC_DIR}/env.sh" ]]; then
    # shellcheck source=./env.sh
    source "${ARC_DIR}/env.sh"
fi
ROOT="${PE_HUB_ROOT:-$(cd "${ARC_DIR}/../../.." && pwd)}"
LOCAL="${ROOT}/services/pe-ensemble/config/training_presets_local"
SHIPPED="${ROOT}/services/pe-ensemble/config/training_presets"
APPLY=0
MODEL="${MODEL:-}"
KEY="${KEY:-}"

for arg in "$@"; do
    case "${arg}" in
        --apply) APPLY=1 ;;
        *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ ! -d "${LOCAL}" ]]; then
    echo "No local presets at ${LOCAL}"
    exit 0
fi

PY="$(command -v python 2>/dev/null || command -v python3)"
APPLY="${APPLY}" MODEL="${MODEL}" KEY="${KEY}" LOCAL="${LOCAL}" SHIPPED="${SHIPPED}" \
"${PY}" - <<'PY'
import os, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required", file=sys.stderr)
    sys.exit(2)

local_dir = Path(os.environ["LOCAL"])
shipped_dir = Path(os.environ["SHIPPED"])
apply = os.environ.get("APPLY") == "1"
model_filter = (os.environ.get("MODEL") or "").strip().lower()
key_filter = (os.environ.get("KEY") or "").strip().lower().replace("-", "_")

files = sorted(local_dir.glob("*.yaml"))
if not files:
    print("No local *.yaml presets to promote.")
    sys.exit(0)

for src in files:
    model = src.stem.lower()
    if model_filter and model != model_filter:
        continue
    bundle = yaml.safe_load(src.read_text()) or {}
    datasets = bundle.get("datasets") or {}
    if not isinstance(datasets, dict) or not datasets:
        print(f"  skip {src.name}: no datasets")
        continue
    if key_filter:
        datasets = {k: v for k, v in datasets.items() if k == key_filter}
        if not datasets:
            print(f"  skip {src.name}: key {key_filter!r} not found")
            continue

    dst = shipped_dir / src.name
    if dst.is_file():
        shipped = yaml.safe_load(dst.read_text()) or {}
    else:
        shipped = {
            "schema_version": 1,
            "model": model,
            "defaults": {},
            "datasets": {},
        }
    if "datasets" not in shipped or not isinstance(shipped["datasets"], dict):
        shipped["datasets"] = {}
    shipped.setdefault("model", model)

    print(f"{src.name}:")
    for k, entry in datasets.items():
        print(f"  + datasets.{k}")
        if apply:
            shipped["datasets"][k] = entry

    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(yaml.safe_dump(shipped, sort_keys=False, default_flow_style=False))
        print(f"  wrote {dst}")

if not apply:
    print("\nDry run. Re-run with --apply to merge into shipped, then git add/commit.")
else:
    print(f"\nMerged. Review and commit under {shipped_dir}")
PY

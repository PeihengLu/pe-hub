#!/usr/bin/env python3
"""Stage preferred scratch-benchmark weights for a GitHub Release asset.

Builds (gitignored under txt/ — do not commit):

  txt/supplementary/scratch-benchmark-weights/
    README.md
    weights_id_map.tsv
    registry_entries.json
    weights/<model>/<id>/

Then pack with:

  ./scripts/experiments/scratch-benchmark/pack_release_weights.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
MAP_PATH = REPO / "scripts/experiments/scratch-benchmark/weights_id_map.tsv"
WEIGHTS_ROOT = REPO / "services/pe-ensemble/weights"
LOCAL_REGISTRY = WEIGHTS_ROOT / "local_registry.json"
OUT_ROOT = REPO / "txt" / "supplementary" / "scratch-benchmark-weights"
OUT_WEIGHTS = OUT_ROOT / "weights"

README = """# Scratch-benchmark preferred weights

From-scratch holdout₃ checkpoints for **DeepPrime**, **OPED**, and **PRIDICT2**
(15 datasheets × 3 seeds = **135** weight sets). Same selection as the Results
heatmaps:

| Model | Source run |
|-------|------------|
| OPED (all cells) | `20260921T071524Z` |
| DeepPrime / PRIDICT2 unpooled Lib-Diverse + Lib-MMR/CV | `20260917T153347Z` |
| DeepPrime / PRIDICT2 single-condition (Library1, ClinVar, DeepPE, MinSePIE) | `20260911T065340Z` |

## Layout

```
weights_id_map.tsv       # model × bench × seed → weights_id + metrics
registry_entries.json    # pe-ensemble local_registry summaries (preferred only)
weights/<model>/<id>/    # trainable weight directories
```

This tree is **not** committed to git. It is shipped as the
`scratch-benchmark-weights.zip` GitHub Release asset.

## Install into pe-ensemble

From the repository root (after downloading the release zip):

```bash
./scripts/experiments/scratch-benchmark/install_release_weights.sh \\
  /path/to/scratch-benchmark-weights.zip
```

Or unpack manually and copy `weights/` into `services/pe-ensemble/weights/`, then:

```bash
python3 scripts/experiments/scratch-benchmark/publish_preferred_weights.py
```
"""


def load_map_rows() -> list[dict[str, str]]:
    if not MAP_PATH.is_file():
        raise SystemExit(
            f"Missing {MAP_PATH}. Run publish_preferred_weights.py first."
        )
    lines = MAP_PATH.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def link_or_copy_tree(src: Path, dst: Path) -> str:
    """Prefer hardlinks (same filesystem); fall back to copy."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        for root, _dirs, files in os.walk(src):
            rel = Path(root).relative_to(src)
            target_root = dst / rel
            target_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                s = Path(root) / name
                d = target_root / name
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy2(s, d)
        return "hardlink"
    except OSError:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return "copy"


def export_registry_entries(weight_ids: set[str]) -> list[dict[str, Any]]:
    if not LOCAL_REGISTRY.is_file():
        return []
    payload = json.loads(LOCAL_REGISTRY.read_text(encoding="utf-8"))
    return [
        entry
        for entry in payload.get("entries") or []
        if isinstance(entry, dict) and entry.get("id") in weight_ids
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-copy",
        action="store_true",
        help="Always copy files instead of hardlinking",
    )
    args = parser.parse_args(argv)

    rows = load_map_rows()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_WEIGHTS.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MAP_PATH, OUT_ROOT / "weights_id_map.tsv")
    (OUT_ROOT / "README.md").write_text(README, encoding="utf-8")

    missing: list[str] = []
    modes: dict[str, int] = {"hardlink": 0, "copy": 0}
    for row in rows:
        model = row["model"]
        wid = row["weights_id"]
        src = WEIGHTS_ROOT / model / wid
        dst = OUT_WEIGHTS / model / wid
        if not src.is_dir():
            missing.append(f"{model}/{wid}")
            continue
        if args.force_copy:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            modes["copy"] += 1
        else:
            mode = link_or_copy_tree(src, dst)
            modes[mode] += 1

    if missing:
        print(
            f"Error: {len(missing)} weight dirs missing under {WEIGHTS_ROOT}",
            file=sys.stderr,
        )
        for item in missing[:10]:
            print(f"  missing: {item}", file=sys.stderr)
        return 1

    weight_ids = {row["weights_id"] for row in rows}
    entries = export_registry_entries(weight_ids)
    (OUT_ROOT / "registry_entries.json").write_text(
        json.dumps(
            {
                "version": 1,
                "count": len(entries),
                "description": "Preferred scratch-benchmark local_registry summaries",
                "entries": sorted(entries, key=lambda e: str(e.get("id") or "")),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    keep = {(row["model"], row["weights_id"]) for row in rows}
    for model_dir in OUT_WEIGHTS.iterdir():
        if not model_dir.is_dir():
            continue
        for entry_dir in list(model_dir.iterdir()):
            if (model_dir.name, entry_dir.name) not in keep:
                shutil.rmtree(entry_dir)

    n_dirs = sum(1 for _ in OUT_WEIGHTS.glob("*/*") if _.is_dir())
    print(f"Staged {n_dirs} weight dirs → {OUT_WEIGHTS}")
    print(f"  hardlink={modes['hardlink']} copy={modes['copy']}")
    print(f"Wrote {OUT_ROOT / 'README.md'}")
    print(f"Wrote {OUT_ROOT / 'weights_id_map.tsv'}")
    print(f"Wrote {OUT_ROOT / 'registry_entries.json'} ({len(entries)} entries)")
    print("Next: ./scripts/experiments/scratch-benchmark/pack_release_weights.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

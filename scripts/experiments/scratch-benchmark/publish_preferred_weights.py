#!/usr/bin/env python3
"""Curate preferred scratch-benchmark weights for pe-ensemble reuse.

Filters the later correct runs (same selection as the Results heatmaps):

  - OPED:            results/20260921T071524Z
  - DeepPrime/PRIDICT2 unpooled cells: 20260917T153347Z
  - Single-condition sheets (Library 1 / ClinVar / DeepPE / MinSePIE):
                     20260911T065340Z

Writes:
  - weights_id_map.tsv   — model × bench × seed → weights_id (+ metrics)
  - weight_artifacts.txt — repo-relative paths for ONLY=scratch-weights pull

Also relabels matching entries in ``services/pe-ensemble/weights/local_registry.json``
(and on-disk manifests when the weight directory is present) so ``peen weights``
shows a stable scratch-benchmark label.

Does **not** call rebuild_index() (that would drop registry rows whose dirs are
still only on ARC). After pulling weight dirs:

  ONLY=scratch-weights ./scripts/cluster/oxford-arc/pull_from_arc.sh <ARC_USER>
  python3 scripts/experiments/scratch-benchmark/publish_preferred_weights.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "scripts/experiments/scratch-benchmark/results"
OUT_DIR = Path(__file__).resolve().parent
WEIGHTS_ROOT = REPO / "services/pe-ensemble/weights"
LOCAL_REGISTRY = WEIGHTS_ROOT / "local_registry.json"

OPED_RUN = "20260921T071524Z"
SPLIT_RUN = "20260917T153347Z"
POOLED_RUN = "20260911T065340Z"

SINGLE_CONDITION = {
    "pridict1-library1",
    "deepprime-clinvar",
    "deeppe-pooled",
    "minsepie-insert-pooled",
}

MODELS = ["deepprime", "oped", "pridict2"]
MODEL_LABELS = {
    "deepprime": "DeepPrime",
    "oped": "OPED",
    "pridict2": "PRIDICT2",
}

# Same 15-cell matrix as txt/diagrams/generate_scratch_heatmap.py
BENCHES = [
    "deeppe-pooled",
    "deepprime-clinvar",
    "pridict1-library1",
    "pridict2-library-diverse__hek293t",
    "pridict2-library-diverse__k562",
    "pridict2-library-diverse__k562mlh1dn",
    "minsepie-insert-pooled",
    "optiprime-lib-mmr__hek293t__pe2",
    "optiprime-lib-mmr__hek293t__pe4",
    "optiprime-lib-mmr__hela__pe2",
    "optiprime-lib-mmr__hela__pe4",
    "optiprime-lib-cv__hek293t__pe2",
    "optiprime-lib-cv__hek293t__pe4",
    "optiprime-lib-cv__hela__pe2",
    "optiprime-lib-cv__hela__pe4",
]


def cell_dir(model: str, bench: str) -> Optional[Path]:
    if model == "oped":
        path = RESULTS / OPED_RUN / f"oped__{bench}"
        return path if path.is_dir() else None
    for run in (SPLIT_RUN, POOLED_RUN):
        if bench not in SINGLE_CONDITION and run == POOLED_RUN:
            continue
        path = RESULTS / run / f"{model}__{bench}"
        if (path / "state").is_dir() and any((path / "state").glob("seed_*.json")):
            return path
    return None


def preferred_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for bench in BENCHES:
            cell = cell_dir(model, bench)
            if cell is None:
                print(f"warn: missing cell {model}__{bench}", file=sys.stderr)
                continue
            run_id = cell.parent.name
            for state_path in sorted((cell / "state").glob("seed_*.json")):
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                if payload.get("status") != "ok":
                    continue
                weights_id = payload.get("weights_id")
                if not weights_id:
                    continue
                seed = payload.get("seed")
                if seed is None:
                    # seed_42.json → 42
                    try:
                        seed = int(state_path.stem.split("_", 1)[1])
                    except (IndexError, ValueError):
                        seed = state_path.stem
                rows.append(
                    {
                        "model": model,
                        "bench": bench,
                        "seed": int(seed) if str(seed).isdigit() else seed,
                        "repeat_id": state_path.stem,
                        "run_id": run_id,
                        "weights_id": str(weights_id),
                        "test_pearson": payload.get("test_pearson"),
                        "test_spearman": payload.get("test_spearman"),
                        "test_mse": payload.get("test_mse"),
                        "best_trial": payload.get("best_trial"),
                        "best_value": payload.get("best_value"),
                        "cell_dir": str(cell.relative_to(REPO)),
                        "weight_dir": f"services/pe-ensemble/weights/{model}/{weights_id}",
                    }
                )
    return rows


def scratch_label(row: dict[str, Any]) -> str:
    model = MODEL_LABELS.get(row["model"], row["model"])
    pearson = row.get("test_pearson")
    r_bit = f", r={pearson:.3f}" if isinstance(pearson, (int, float)) else ""
    return (
        f"Scratch holdout3 | {model} | {row['bench']} | "
        f"{row['repeat_id']}{r_bit}"
    )


def scratch_notes(row: dict[str, Any]) -> str:
    return (
        f"scratch-benchmark preferred "
        f"run={row['run_id']} model={row['model']} bench={row['bench']} "
        f"{row['repeat_id']}"
    )


def write_weights_id_map(rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        "model",
        "bench",
        "seed",
        "repeat_id",
        "run_id",
        "weights_id",
        "test_pearson",
        "test_spearman",
        "test_mse",
        "best_trial",
        "best_value",
        "weight_dir",
    ]
    lines = ["\t".join(cols)]
    for row in rows:
        lines.append("\t".join("" if row.get(c) is None else str(row[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_weight_artifacts(rows: list[dict[str, Any]], path: Path) -> None:
    weight_dirs = sorted({row["weight_dir"] for row in rows})
    cell_dirs = sorted({row["cell_dir"] for row in rows})
    lines = [
        "# Repo-relative paths for the scratch-benchmark preferred-weight bundle.",
        "# Used by: ONLY=scratch-weights ./scripts/cluster/oxford-arc/pull_from_arc.sh",
        "# and by publish_preferred_weights.py",
        "#",
        f"# {len(rows)} seed checkpoints across {len(cell_dirs)} cells "
        f"({len(weight_dirs)} weight dirs).",
        "#",
        f"# Selection: OPED→{OPED_RUN}; DeepPrime/PRIDICT2 unpooled→"
        f"{SPLIT_RUN};",
        f"# single-condition (Library1/ClinVar/DeepPE/MinSePIE)→{POOLED_RUN}.",
        "",
        "scripts/experiments/scratch-benchmark/weights_id_map.tsv",
        "scripts/experiments/scratch-benchmark/weight_artifacts.txt",
        "services/pe-ensemble/weights/local_registry.json",
        "",
        "# Preferred result cells (state JSON + metrics)",
    ]
    for cell in cell_dirs:
        lines.append(cell)
    lines.append("")
    lines.append("# Trained weight directories")
    for wdir in weight_dirs:
        lines.append(wdir)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relabel_registry(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Update local_registry + on-disk manifests. Returns (reg, disk, missing)."""
    if not LOCAL_REGISTRY.is_file():
        print(f"warn: no {LOCAL_REGISTRY}", file=sys.stderr)
        return 0, 0, len(rows)

    payload = json.loads(LOCAL_REGISTRY.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    by_id = {e["id"]: e for e in entries if isinstance(e, dict) and "id" in e}

    updated_reg = 0
    updated_disk = 0
    missing = 0
    for row in rows:
        wid = row["weights_id"]
        label = scratch_label(row)
        notes = scratch_notes(row)
        entry = by_id.get(wid)
        if entry is not None:
            changed = entry.get("label") != label or entry.get("notes") != notes
            entry["label"] = label
            entry["notes"] = notes
            # Keep a small provenance blob for later filtering.
            meta = dict(entry.get("scratch_benchmark") or {})
            meta.update(
                {
                    "run_id": row["run_id"],
                    "bench": row["bench"],
                    "seed": row["seed"],
                    "repeat_id": row["repeat_id"],
                    "preferred": True,
                }
            )
            entry["scratch_benchmark"] = meta
            if changed:
                updated_reg += 1
        else:
            # Metadata-only stub so the id is discoverable before the dir is pulled.
            entries.append(
                {
                    "id": wid,
                    "model": row["model"],
                    "label": label,
                    "source": "trained",
                    "format": None,
                    "created_at": None,
                    "metrics": {
                        "test": {
                            "pearson": row.get("test_pearson"),
                            "spearman": row.get("test_spearman"),
                            "mse": row.get("test_mse"),
                        }
                    },
                    "notes": notes,
                    "scratch_benchmark": {
                        "run_id": row["run_id"],
                        "bench": row["bench"],
                        "seed": row["seed"],
                        "repeat_id": row["repeat_id"],
                        "preferred": True,
                        "pending_pull": True,
                    },
                }
            )
            by_id[wid] = entries[-1]
            updated_reg += 1

        weight_dir = WEIGHTS_ROOT / row["model"] / wid
        manifest_path = weight_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["label"] = label
            manifest["notes"] = notes
            manifest["scratch_benchmark"] = by_id[wid]["scratch_benchmark"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            updated_disk += 1
        else:
            missing += 1

    payload["entries"] = entries
    payload["count"] = len(entries)
    LOCAL_REGISTRY.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return updated_reg, updated_disk, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-relabel",
        action="store_true",
        help="Only rewrite map/artifacts; do not touch local_registry",
    )
    parser.add_argument(
        "--no-write-map",
        action="store_true",
        help="Skip rewriting weights_id_map.tsv / weight_artifacts.txt",
    )
    args = parser.parse_args(argv)

    rows = preferred_rows()
    if not rows:
        print("Error: no preferred ok seeds found", file=sys.stderr)
        return 1

    map_path = OUT_DIR / "weights_id_map.tsv"
    art_path = OUT_DIR / "weight_artifacts.txt"
    if not args.no_write_map:
        write_weights_id_map(rows, map_path)
        write_weight_artifacts(rows, art_path)
        print(f"Wrote {map_path} ({len(rows)} rows)")
        print(f"Wrote {art_path}")

    on_disk = sum(
        1 for row in rows if (WEIGHTS_ROOT / row["model"] / row["weights_id"]).is_dir()
    )
    print(f"Preferred ok seeds: {len(rows)}; weight dirs on disk: {on_disk}/{len(rows)}")

    if not args.skip_relabel:
        n_reg, n_disk, n_miss = relabel_registry(rows)
        print(
            f"Registry labels updated: {n_reg}; "
            f"on-disk manifests: {n_disk}; "
            f"dirs still missing (pull from ARC): {n_miss}"
        )
        if n_miss:
            print(
                "Pull weight blobs:\n"
                "  ONLY=scratch-weights ./scripts/cluster/oxford-arc/pull_from_arc.sh <ARC_USER>\n"
                "Then re-run this script to stamp on-disk manifests."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

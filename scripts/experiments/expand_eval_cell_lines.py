#!/usr/bin/env python3
"""Expand eval benchmark specs so each cell line is scored separately.

Reads ``name|study|dataset1,dataset2,...`` lines and prints
``name|study|datasets|cell_line`` (``name`` gains a ``__{cell}`` suffix when a
benchmark spans more than one cell line).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pe_common.cell_lines import canonical_cell_line


def _normalize_segment(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def cell_lines_for_datasets(
    datasets_dir: Path,
    study: str,
    datasets: list[str],
) -> list[str]:
    """Return sorted cell-line stems from standardized ``{cell}-{pe}.parquet`` files."""
    study_key = _normalize_segment(study)
    found: set[str] = set()
    for dataset in datasets:
        folder = datasets_dir / "standardized" / study_key / _normalize_segment(dataset)
        if not folder.is_dir():
            continue
        for path in folder.glob("*.parquet"):
            stem = path.stem
            if "-" not in stem:
                continue
            cell, _pe = stem.rsplit("-", 1)
            if cell:
                found.add(canonical_cell_line(cell))
    return sorted(found)


def expand_benchmark_spec(spec: str, datasets_dir: Path) -> list[str]:
    """Expand one ``name|study|datasets`` spec into per-cell-line specs."""
    parts = spec.split("|")
    if len(parts) < 3:
        raise ValueError(f"Benchmark spec must be name|study|datasets, got: {spec!r}")
    name, study, datasets_csv = parts[0], parts[1], parts[2]
    datasets = [item for item in datasets_csv.split(",") if item.strip()]
    cells = cell_lines_for_datasets(datasets_dir, study, datasets)
    if not cells:
        return [f"{name}|{study}|{datasets_csv}|"]
    if len(cells) == 1:
        return [f"{name}|{study}|{datasets_csv}|{cells[0]}"]
    return [f"{name}__{cell}|{study}|{datasets_csv}|{cell}" for cell in cells]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        required=True,
        help="Repository datasets/ directory (contains standardized/)",
    )
    parser.add_argument(
        "specs",
        nargs="+",
        help="Benchmark specs: name|study|dataset[,dataset...]",
    )
    args = parser.parse_args(argv)
    for spec in args.specs:
        for expanded in expand_benchmark_spec(spec, args.datasets_dir):
            print(expanded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

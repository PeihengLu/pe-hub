#!/usr/bin/env python3
"""Expand eval benchmark specs so each cell line (and PE system) is scored separately.

Reads ``name|study|dataset1,dataset2,...`` lines and prints
``name|study|datasets|cell_line|pe_system``.

``name`` gains a ``__{cell}`` suffix when a benchmark spans more than one cell
line, and a ``__{pe}`` suffix when it spans more than one PE system (OptiPrime
Lib-MMR / Lib-CV PE2 vs PE4). Single-PE benches keep their current names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pe_common.cell_lines import canonical_cell_line


def _normalize_segment(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_pe_system(value: str) -> str:
    return str(value).strip().lower().replace("-", "")


def conditions_for_datasets(
    datasets_dir: Path,
    study: str,
    datasets: list[str],
) -> list[tuple[str, str]]:
    """Return sorted ``(cell_line, pe_system)`` from standardized parquet stems."""
    study_key = _normalize_segment(study)
    found: set[tuple[str, str]] = set()
    for dataset in datasets:
        folder = datasets_dir / "standardized" / study_key / _normalize_segment(dataset)
        if not folder.is_dir():
            continue
        for path in folder.glob("*.parquet"):
            stem = path.stem
            if "-" not in stem:
                continue
            cell, pe = stem.rsplit("-", 1)
            if cell and pe:
                found.add((canonical_cell_line(cell), _normalize_pe_system(pe)))
    return sorted(found)


def cell_lines_for_datasets(
    datasets_dir: Path,
    study: str,
    datasets: list[str],
) -> list[str]:
    """Return sorted cell-line stems from standardized ``{cell}-{pe}.parquet`` files."""
    return sorted({cell for cell, _pe in conditions_for_datasets(datasets_dir, study, datasets)})


def expand_benchmark_spec(spec: str, datasets_dir: Path) -> list[str]:
    """Expand one ``name|study|datasets`` spec into per-cell / per-PE specs."""
    parts = spec.split("|")
    if len(parts) < 3:
        raise ValueError(f"Benchmark spec must be name|study|datasets, got: {spec!r}")
    name, study, datasets_csv = parts[0], parts[1], parts[2]
    datasets = [item for item in datasets_csv.split(",") if item.strip()]
    conditions = conditions_for_datasets(datasets_dir, study, datasets)
    if not conditions:
        return [f"{name}|{study}|{datasets_csv}||"]
    cells = {cell for cell, _pe in conditions}
    pes = {pe for _cell, pe in conditions}
    out: list[str] = []
    for cell, pe in conditions:
        bench = name
        if len(cells) > 1:
            bench = f"{bench}__{cell}"
        if len(pes) > 1:
            bench = f"{bench}__{pe}"
        out.append(f"{bench}|{study}|{datasets_csv}|{cell}|{pe}")
    return out


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

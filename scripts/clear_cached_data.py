#!/usr/bin/env python3
"""Remove derived PE-DB caches so standardize/convert rebuild from current code.

Does not touch ``datasets/raw/``, ``exported/``, ``catalog/``, or reference genomes.

Usage:
  python scripts/clear_cached_data.py
  python scripts/clear_cached_data.py --dry-run
  python scripts/clear_cached_data.py --standardized
  python scripts/clear_cached_data.py --all
  python scripts/clear_cached_data.py --study pridict2 --format pridict2
  python scripts/clear_cached_data.py --all --study optiprime

After converter or standardizer changes, clear both layers then rebuild::

  python scripts/clear_cached_data.py --all
  pedb standardize --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services" / "pe-db"))
sys.path.insert(0, str(REPO / "packages" / "pe-common"))

from app.formatted_cache import clear_cached_data  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clear PE-DB formatted and/or standardized caches.",
    )
    parser.add_argument(
        "--formatted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear datasets/formatted (model-format parquet, default on).",
    )
    parser.add_argument(
        "--standardized",
        action="store_true",
        help="Also clear datasets/standardized parquet.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear formatted and standardized caches.",
    )
    parser.add_argument("--study", default=None, help="Limit to one study key.")
    parser.add_argument(
        "--format",
        dest="target_format",
        default=None,
        help="Limit formatted cache to one model format (pridict2, deepprime, …).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help="Override datasets/ root (default: pe_common DATA_ROOT).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    formatted = True if args.all else bool(args.formatted)
    standardized = True if args.all else bool(args.standardized)
    if not formatted and not standardized:
        print("Nothing to clear: pass --formatted and/or --standardized (or --all).", file=sys.stderr)
        return 2
    report = clear_cached_data(
        formatted=formatted,
        standardized=standardized,
        study=args.study,
        target_format=args.target_format,
        datasets_dir=args.datasets_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    n_files = sum(int(row["files"]) for row in report["targets"])
    n_bytes = sum(int(row["bytes"]) for row in report["targets"])
    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {len(report['targets'])} cache root(s), {n_files} files, {n_bytes} bytes.")
    if standardized and not args.dry_run:
        print("Re-standardize with: pedb standardize --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

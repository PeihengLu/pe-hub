#!/usr/bin/env python3
"""Command-line interface for PE Database (init, export, filter) without HTTP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from pe_db.library import (
    PeDbLibraryError,
    catalog_statistics,
    filter_data,
    list_datasheets,
    list_datasets,
    list_output_formats,
    list_scaffolds,
    list_studies,
    reload_plugins,
    run_convert_sheet,
    run_export,
    run_init,
    run_seed,
    run_standardize,
)


def _payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(group["records"])
        for group in payload.get("groups", [])
        if group.get("records")
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_filter_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("filter", help="Filter catalog and/or export model-format data")
    parser.add_argument("--format", dest="format_", default=None, help="Output format (deepprime, oped, std, …)")
    parser.add_argument("--split-strategy", default=None, choices=["none", "holdout_2", "holdout_3", "cv"])
    parser.add_argument("--train-pct", type=float, default=None)
    parser.add_argument("--val-pct", type=float, default=None)
    parser.add_argument("--test-pct", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--use-original-fold", action="store_true")
    parser.add_argument("--original-fold-test-value", type=float, default=-1.0)
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    for name in (
        "study",
        "dataset",
        "cell-line",
        "pe-system",
        "edit-type",
        "edit-scope",
        "experimental-method",
        "target-context",
        "scaffold-name",
    ):
        parser.add_argument(
            f"--{name}",
            action="append",
            default=[],
            dest=name.replace("-", "_"),
        )
    parser.add_argument("--edit-length", action="append", type=int, default=[])
    parser.add_argument("--edit-efficiency-min", type=float, default=None)
    parser.add_argument("--edit-efficiency-max", type=float, default=None)
    parser.add_argument(
        "--out",
        default=None,
        help="Write export to file (.json full payload, .csv/.parquet merged rows)",
    )
    parser.set_defaults(func=cmd_filter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pe-db",
        description="PE Database headless tools (catalog init, export, model-format filter).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Seed catalog, export raw data, and standardize")
    init_p.add_argument("--force-export", action="store_true")
    init_p.add_argument("--force-standardize", action="store_true")
    init_p.set_defaults(func=cmd_init)

    sub.add_parser("seed", help="Seed catalog tables only").set_defaults(func=cmd_seed)

    export_p = sub.add_parser("export", help="Export raw study files (and optionally standardize)")
    export_p.add_argument("--study", default=None)
    export_p.add_argument("--force-reexport", action="store_true")
    export_p.add_argument("--no-standardize", action="store_true")
    export_p.add_argument("--force-standardize", action="store_true")
    export_p.set_defaults(func=cmd_export)

    std_p = sub.add_parser("standardize", help="Standardize exported CSVs to parquet")
    std_p.add_argument("--study", default=None)
    std_p.add_argument("--force", action="store_true")
    std_p.set_defaults(func=cmd_standardize)

    convert_p = sub.add_parser("convert", help="Standardize one exported datasheet")
    convert_p.add_argument("--study", required=True)
    convert_p.add_argument("--dataset", required=True)
    convert_p.add_argument("--cell-line", required=True)
    convert_p.add_argument("--pe-system", required=True)
    convert_p.set_defaults(func=cmd_convert)

    _build_filter_parser(sub)

    studies_p = sub.add_parser("studies", help="List catalog studies")
    studies_p.set_defaults(func=cmd_studies)

    datasets_p = sub.add_parser("datasets", help="List catalog datasets")
    datasets_p.add_argument("--study", default=None)
    datasets_p.set_defaults(func=cmd_datasets)

    datasheets_p = sub.add_parser("datasheets", help="List catalog datasheets")
    datasheets_p.add_argument("--study", default=None)
    datasheets_p.add_argument("--dataset", default=None)
    datasheets_p.set_defaults(func=cmd_datasheets)

    scaffolds_p = sub.add_parser("scaffolds", help="List pegRNA scaffolds")
    scaffolds_p.set_defaults(func=cmd_scaffolds)

    stats_p = sub.add_parser("statistics", help="Descriptive statistics over edit rows")
    stats_p.add_argument("--edit-type", default=None)
    stats_p.add_argument("--edit-length", type=int, default=None)
    stats_p.add_argument("--edit-efficiency-min", type=float, default=None)
    stats_p.add_argument("--edit-efficiency-max", type=float, default=None)
    stats_p.add_argument("--edit-scope", default=None)
    stats_p.add_argument("--experimental-method", default=None)
    stats_p.add_argument("--target-context", default=None)
    stats_p.add_argument("--scaffold-name", default=None)
    stats_p.set_defaults(func=cmd_statistics)

    formats_p = sub.add_parser("formats", help="List supported filter output formats")
    formats_p.set_defaults(func=cmd_formats)

    plugins_p = sub.add_parser("plugins", help="Plugin management")
    plugins_sub = plugins_p.add_subparsers(dest="plugins_command", required=True)
    plugins_sub.add_parser("reload", help="Reload active plugin converters").set_defaults(
        func=cmd_plugins_reload
    )

    return parser


def cmd_init(args: argparse.Namespace) -> int:
    run_init(force_export=args.force_export, force_standardize=args.force_standardize)
    print("PE Database initialization complete.")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    del args
    run_seed()
    print("Catalog seeded.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    result = run_export(
        study=args.study,
        force_reexport=args.force_reexport,
        standardize=not args.no_standardize,
        force_standardize=args.force_standardize,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_standardize(args: argparse.Namespace) -> int:
    result = run_standardize(study=args.study, force=args.force)
    print(json.dumps(result, indent=2))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    result = run_convert_sheet(
        study=args.study,
        dataset=args.dataset,
        cell_line=args.cell_line,
        pe_system=args.pe_system,
    )
    print(json.dumps(result, indent=2))
    return 0


def _optional_list(values: List[Any]) -> Optional[List[Any]]:
    return values or None


def cmd_filter(args: argparse.Namespace) -> int:
    result = filter_data(
        study=_optional_list(args.study),
        dataset=_optional_list(args.dataset),
        cell_line=_optional_list(args.cell_line),
        pe_system=_optional_list(args.pe_system),
        edit_type=_optional_list(args.edit_type),
        edit_length=_optional_list(args.edit_length),
        edit_efficiency_min=args.edit_efficiency_min,
        edit_efficiency_max=args.edit_efficiency_max,
        edit_scope=_optional_list(args.edit_scope),
        experimental_method=_optional_list(args.experimental_method),
        target_context=_optional_list(args.target_context),
        scaffold_name=_optional_list(args.scaffold_name),
        format_=args.format_,
        split_strategy=args.split_strategy,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        cv_folds=args.cv_folds,
        use_original_fold=args.use_original_fold,
        original_fold_test_value=args.original_fold_test_value,
        split_random_state=args.split_random_state,
        merge=args.merge,
        summary_only=args.summary_only,
    )
    if args.out:
        _write_filter_output(result, Path(args.out))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


def _write_filter_output(result: dict[str, Any], path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {path}")
        return
    if suffix in {".csv", ".parquet"}:
        df = _payload_to_dataframe(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        print(f"Wrote {len(df)} rows to {path}")
        return
    raise PeDbLibraryError(f"Unsupported --out extension: {path.suffix} (use .json, .csv, or .parquet)")


def cmd_studies(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(list_studies(), indent=2, default=str))
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    print(json.dumps(list_datasets(study=args.study), indent=2, default=str))
    return 0


def cmd_datasheets(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            list_datasheets(study=args.study, dataset=args.dataset),
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_scaffolds(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(list_scaffolds(), indent=2, default=str))
    return 0


def cmd_statistics(args: argparse.Namespace) -> int:
    result = catalog_statistics(
        edit_type=args.edit_type,
        edit_length=args.edit_length,
        edit_efficiency_min=args.edit_efficiency_min,
        edit_efficiency_max=args.edit_efficiency_max,
        edit_scope=args.edit_scope,
        experimental_method=args.experimental_method,
        target_context=args.target_context,
        scaffold_name=args.scaffold_name,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_formats(args: argparse.Namespace) -> int:
    del args
    for name in list_output_formats():
        print(name)
    return 0


def cmd_plugins_reload(args: argparse.Namespace) -> int:
    del args
    loaded = reload_plugins()
    print(json.dumps({"loaded": loaded, "count": len(loaded)}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PeDbLibraryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

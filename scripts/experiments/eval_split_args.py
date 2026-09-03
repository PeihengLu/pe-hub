#!/usr/bin/env python3
"""Split flags for ``evaluate_base_model_benchmarks.sh``.

Vendor PRIDICT2 ``run_x`` is the library-diverse CV fold that held out
``testset_fold == x`` (copied to PE-DB ``original_fold``). Evaluating that
checkpoint on the other folds scores training loci.

PRIDICT library1 has no author split, so vendor training used every locus.
A random holdout on that sheet is only a scoring convenience; leak checks
must still treat the full library1 locus set as train.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence


def _norm(value: str) -> str:
    return str(value).strip().lower()


def _datasets_list(datasets: Sequence[str] | str) -> list[str]:
    if isinstance(datasets, str):
        items = datasets.split(",")
    else:
        items = list(datasets)
    return [_norm(item) for item in items if str(item).strip()]


def evaluation_split_cli_args(
    *,
    model: str,
    study: str,
    datasets: Sequence[str] | str,
    cv_run: Optional[int] = None,
) -> list[str]:
    """Return peen ``evaluate`` split flags for one matrix cell."""
    model_key = _norm(model)
    study_key = _norm(study)
    dataset_keys = _datasets_list(datasets)

    if study_key == "deepprime" and dataset_keys == ["deepprime-clinvar"]:
        return ["--use-original-fold"]

    if (
        model_key == "pridict2"
        and study_key == "pridict2"
        and "library-diverse" in dataset_keys
        and cv_run is not None
    ):
        return [
            "--use-original-fold",
            "--original-fold-test-value",
            str(int(cv_run)),
        ]

    return ["--no-use-original-fold"]


def original_fold_test_value_from_args(args: Sequence[str]) -> Optional[int]:
    """Parse ``--original-fold-test-value`` from split CLI args, if present."""
    tokens = list(args)
    for index, token in enumerate(tokens):
        if token == "--original-fold-test-value" and index + 1 < len(tokens):
            return int(float(tokens[index + 1]))
        if token.startswith("--original-fold-test-value="):
            return int(float(token.split("=", 1)[1]))
    return None


def eval_result_cell_key(
    model: Any = None,
    weights: Any = None,
    benchmark_name: Any = None,
    cell_line: Any = None,
    original_fold_test_value: Any = None,
    **_ignored: Any,
) -> str:
    """Identity for skip-existing / jsonl compaction.

    Fold-matched library-diverse cells include ``fold:x`` so a prior random
    holdout of the same weight does not skip the author-fold rerun.
    """
    parts = [
        str(model or ""),
        str(weights or ""),
        str(benchmark_name or ""),
        str(cell_line or ""),
    ]
    if original_fold_test_value is not None and original_fold_test_value != "":
        parts.append(f"fold:{int(float(original_fold_test_value))}")
    return "|".join(parts)


def eval_result_cell_key_from_record(record: dict[str, Any]) -> str:
    return eval_result_cell_key(
        model=record.get("model"),
        weights=record.get("weights"),
        benchmark_name=record.get("benchmark_name"),
        cell_line=record.get("cell_line"),
        original_fold_test_value=record.get("original_fold_test_value"),
    )


def split_plan(
    *,
    model: str,
    study: str,
    datasets: Sequence[str] | str,
    cv_run: Optional[int] = None,
) -> dict[str, Any]:
    args = evaluation_split_cli_args(
        model=model,
        study=study,
        datasets=datasets,
        cv_run=cv_run,
    )
    return {
        "args": args,
        "use_original_fold": "--use-original-fold" in args,
        "original_fold_test_value": original_fold_test_value_from_args(args),
    }


def _parse_cv_run(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return int(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names (e.g. library-diverse)",
    )
    parser.add_argument("--cv-run", default="", help="Vendor run index; empty if none")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print {args, use_original_fold, original_fold_test_value}",
    )
    args = parser.parse_args(argv)
    plan = split_plan(
        model=args.model,
        study=args.study,
        datasets=args.datasets,
        cv_run=_parse_cv_run(args.cv_run),
    )
    if args.json:
        json.dump(plan, sys.stdout)
        sys.stdout.write("\n")
    else:
        for token in plan["args"]:
            print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

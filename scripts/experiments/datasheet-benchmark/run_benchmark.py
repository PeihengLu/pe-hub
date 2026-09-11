#!/usr/bin/env python3
"""From-scratch datasheet benchmark: nested Optuna on N-fold CV or holdout_3 × N seeds.

Uses current ``pedb`` / ``peen`` libraries (``pe_db.library`` + ``pe_ensemble``
tune/train/evaluate). Size chooses the protocol unless ``--protocol`` is set.

  small  (n_rows < threshold)  N-fold CV; Optuna X trials on each outer fold
  large  (n_rows >= threshold) holdout_3 repeated N times (split + init seeds);
                               Optuna X trials on each repeat

N (``--n``) and X (``--n-trials`` / ``-x``) are required user inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from protocol import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_HOLDOUT_TEST_PCT,
    DEFAULT_HOLDOUT_TRAIN_PCT,
    DEFAULT_HOLDOUT_VAL_PCT,
    DEFAULT_INNER_TRAIN_PCT,
    DEFAULT_INNER_VAL_PCT,
    DEFAULT_SIZE_THRESHOLD,
    ProtocolError,
    assign_cv_folds,
    assign_holdout_3,
    choose_protocol,
    extract_eval_metrics,
    fold_labels,
    remap_cv_fold_to_nested_holdout,
    repeat_seeds,
    split_counts,
    summarize_repeats,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.replace([float("inf"), float("-inf")], pd.NA).to_dict(orient="records")
    return [
        {column: _json_safe(cell) for column, cell in record.items()}
        for record in records
    ]


def _optional_list(values: list[Any]) -> Optional[list[Any]]:
    return values or None


def _slug(parts: list[str]) -> str:
    cleaned = [part.strip().lower().replace(" ", "-") for part in parts if part.strip()]
    return "__".join(cleaned) if cleaned else "custom"


def filter_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "study": _optional_list(args.study),
        "dataset": _optional_list(args.dataset),
        "cell_line": _optional_list(args.cell_line),
        "pe_system": _optional_list(args.pe_system),
        "edit_type": _optional_list(args.edit_type),
        "edit_length": _optional_list(args.edit_length),
        "edit_scope": _optional_list(args.edit_scope),
        "experimental_method": _optional_list(args.experimental_method),
        "target_context": _optional_list(args.target_context),
        "scaffold_name": _optional_list(args.scaffold_name),
        "edit_efficiency_min": args.edit_efficiency_min,
        "edit_efficiency_max": args.edit_efficiency_max,
    }


def dataset_label(args: argparse.Namespace) -> str:
    if args.dataset_name:
        return args.dataset_name
    study = ",".join(args.study) if args.study else "study"
    dataset = ",".join(args.dataset) if args.dataset else "dataset"
    edits = ",".join(args.edit_type) if args.edit_type else "all-edits"
    cell = ",".join(args.cell_line) if args.cell_line else ""
    pe = ",".join(args.pe_system) if args.pe_system else ""
    return _slug([study, dataset, cell, pe, edits])


def bootstrap_peen() -> None:
    """Match ``peen`` CLI: in-process PE-DB + active plugins."""
    from pe_ensemble.plugin_loader import load_active_plugins
    from pe_ensemble.training.config import enable_cli_pe_db_access
    from pe_ensemble.training.pe_db_access import PeDbAccessError, reload_pe_db_plugins

    enable_cli_pe_db_access()
    loaded = load_active_plugins()
    if loaded:
        print(f"Loaded plugins: {', '.join(loaded)}", file=sys.stderr)
    try:
        reload_pe_db_plugins()
    except PeDbAccessError:
        pass


def seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import lightning.pytorch as pl

        pl.seed_everything(int(seed), workers=True)
    except ImportError:
        pass


def probe_n_rows(filter_kwargs: dict[str, Any], *, merge: bool) -> tuple[int, dict[str, Any]]:
    from pe_db.library import filter_data

    payload = filter_data(
        **filter_kwargs,
        format_="std",
        split_strategy="none",
        merge=merge,
        summary_only=True,
    )
    n_rows = int(payload.get("total_records") or 0)
    return n_rows, payload


def fetch_model_format_frame(
    *,
    model_format: str,
    filter_kwargs: dict[str, Any],
    merge: bool,
) -> pd.DataFrame:
    from pe_db.library import filter_data

    payload = filter_data(
        **filter_kwargs,
        format_=model_format,
        split_strategy="none",
        merge=merge,
        summary_only=False,
    )
    frames = [
        pd.DataFrame(group["records"])
        for group in payload.get("groups", [])
        if group.get("records")
    ]
    if not frames:
        skipped = payload.get("skipped") or []
        detail = "; ".join(
            f"{entry.get('study')}/{entry.get('dataset')}: {entry.get('reason')}"
            for entry in skipped
        )
        raise ProtocolError(
            "No model-format rows returned"
            + (f" ({detail})" if detail else "")
        )
    return pd.concat(frames, ignore_index=True)


def fixed_hyperparameters_for_model(
    model: str,
    extra: Optional[dict[str, Any]] = None,
    *,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    hp: dict[str, Any] = {"load_pretrained": False}
    if model.strip().lower() == "pridict2":
        hp["loss_func"] = "MSEloss"
        hp["y_ref"] = ["averageedited"]
    if extra:
        hp.update(extra)
    if seed is not None:
        hp["seed"] = int(seed)
    return hp


def _training_request(
    *,
    model: str,
    dataset_name: str,
    filter_kwargs: dict[str, Any],
    records: list[dict[str, Any]],
    hyperparameters: dict[str, Any],
    device: str,
    notes: str,
):
    from pe_ensemble.training.schemas import SplitQueryParams, TrainingRequest

    return TrainingRequest(
        model_name=model,
        dataset_source="pe-db",
        dataset_name=dataset_name,
        hyperparameters=hyperparameters,
        hyperparameter_mode="replace",
        split=SplitQueryParams(split_strategy="none"),
        study=filter_kwargs.get("study"),
        dataset=filter_kwargs.get("dataset"),
        cell_line=filter_kwargs.get("cell_line"),
        pe_system=filter_kwargs.get("pe_system"),
        edit_type=filter_kwargs.get("edit_type"),
        edit_length=filter_kwargs.get("edit_length"),
        edit_scope=filter_kwargs.get("edit_scope"),
        experimental_method=filter_kwargs.get("experimental_method"),
        target_context=filter_kwargs.get("target_context"),
        scaffold_name=filter_kwargs.get("scaffold_name"),
        edit_efficiency_min=filter_kwargs.get("edit_efficiency_min"),
        edit_efficiency_max=filter_kwargs.get("edit_efficiency_max"),
        records=records,
        notes=notes,
        device=device,
    )


def evaluate_assigned(
    *,
    model: str,
    dataset_name: str,
    repeat_id: str,
    weights_id: str,
    assigned: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    from pe_ensemble.evaluation.schemas import EvaluationRequest
    from pe_ensemble.training.schemas import SplitQueryParams
    from pe_ensemble.library import execute_evaluation

    records = dataframe_to_records(assigned)
    eval_request = EvaluationRequest(
        model_name=model,
        benchmark_name=f"{dataset_name}__{repeat_id}",
        weights=str(weights_id),
        split=SplitQueryParams(
            split_strategy="holdout_3",
            train_pct=DEFAULT_HOLDOUT_TRAIN_PCT,
            val_pct=DEFAULT_HOLDOUT_VAL_PCT,
            test_pct=DEFAULT_HOLDOUT_TEST_PCT,
        ),
        records=records,
        device=device,
        auto_training_benchmark=False,
        allow_data_leak=True,
    )
    print(f"  evaluate {repeat_id}: weights={weights_id}", flush=True)
    eval_payload = execute_evaluation(eval_request, device_id=device)
    metrics = extract_eval_metrics(eval_payload)
    status = "ok"
    if eval_payload.get("skipped"):
        status = "skipped"
    elif eval_payload.get("error_type") or eval_payload.get("status") == "error":
        status = "error"
    elif eval_payload.get("metrics") is None:
        status = "unknown"
    return {
        "n_samples": eval_payload.get("n_samples"),
        "test_spearman": metrics["spearman"],
        "test_pearson": metrics["pearson"],
        "test_mse": metrics["mse"],
        "status": status,
        "eval": eval_payload,
    }


def run_tune_and_eval(
    *,
    model: str,
    dataset_name: str,
    filter_kwargs: dict[str, Any],
    assigned: pd.DataFrame,
    n_trials: int,
    device: str,
    seed: int,
    repeat_id: str,
    study_name: str,
    dataset_preset_key: str,
    notes: str,
    no_write_preset: bool,
    extra_hyperparameters: Optional[dict[str, Any]] = None,
    skip_eval: bool = False,
) -> dict[str, Any]:
    from pe_ensemble.training.tuning_schemas import TuningRequest
    from pe_ensemble.library import execute_tuning

    seed_all(seed)
    records = dataframe_to_records(assigned)
    training = _training_request(
        model=model,
        dataset_name=f"{dataset_name}__{repeat_id}",
        filter_kwargs=filter_kwargs,
        records=records,
        hyperparameters=fixed_hyperparameters_for_model(
            model, extra_hyperparameters, seed=seed
        ),
        device=device,
        notes=notes,
    )
    tune_request = TuningRequest(
        training=training,
        n_trials=int(n_trials),
        study_name=study_name,
        dataset_preset_key=dataset_preset_key,
        no_write_preset=no_write_preset,
        register_best_weights=True,
    )
    print(
        f"  tune {repeat_id}: {n_trials} trials, seed={seed}, "
        f"splits={split_counts(assigned)}",
        flush=True,
    )
    tune_summary = execute_tuning(tune_request, device_id=device)
    final_training = tune_summary.get("final_training") or {}
    weights_id = final_training.get("weights_id")
    if not weights_id:
        raise ProtocolError(f"Tuning {repeat_id} finished without registered weights_id")

    if skip_eval:
        print(f"  skip-eval {repeat_id}: weights={weights_id}", flush=True)
        return {
            "repeat_id": repeat_id,
            "seed": seed,
            "weights_id": weights_id,
            "study_name": tune_summary.get("study_name"),
            "best_trial": tune_summary.get("best_trial"),
            "best_value": tune_summary.get("best_value"),
            "best_hyperparameters": tune_summary.get("best_hyperparameters"),
            "preset_path": tune_summary.get("preset_path"),
            "n_samples": None,
            "test_spearman": None,
            "test_pearson": None,
            "test_mse": None,
            "status": "tuned",
            "eval": None,
            "split_counts": split_counts(assigned),
        }

    eval_block = evaluate_assigned(
        model=model,
        dataset_name=dataset_name,
        repeat_id=repeat_id,
        weights_id=str(weights_id),
        assigned=assigned,
        device=device,
    )
    return {
        "repeat_id": repeat_id,
        "seed": seed,
        "weights_id": weights_id,
        "study_name": tune_summary.get("study_name"),
        "best_trial": tune_summary.get("best_trial"),
        "best_value": tune_summary.get("best_value"),
        "best_hyperparameters": tune_summary.get("best_hyperparameters"),
        "preset_path": tune_summary.get("preset_path"),
        "split_counts": split_counts(assigned),
        **eval_block,
    }


def load_state_rows(state_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not state_dir.is_dir():
        return rows
    for path in sorted(state_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def merge_repeat_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    for group in groups:
        for row in group:
            repeat_id = row.get("repeat_id")
            if repeat_id is None:
                unmatched.append(row)
                continue
            by_id[str(repeat_id)] = row
    merged = [by_id[key] for key in sorted(by_id)]
    merged.extend(unmatched)
    return merged


def write_outputs(
    *,
    out_dir: Path,
    run_id: str,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    state_dir: Optional[Path] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".write.lock"
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if state_dir is not None:
            rows = merge_repeat_rows(rows, load_state_rows(state_dir))
        jsonl_path = out_dir / "results.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        summary_csv = out_dir / "summary.csv"
        fields = [
            "run_id",
            "model",
            "protocol",
            "repeat_id",
            "seed",
            "n_trials",
            "weights_id",
            "status",
            "n_samples",
            "test_spearman",
            "test_pearson",
            "test_mse",
            "best_value",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**plan, **row, "run_id": run_id})

        agg = summarize_repeats(rows)
        agg_path = out_dir / "summary_mean_std.csv"
        with agg_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(agg.keys()))
            writer.writeheader()
            writer.writerow(agg)

        plan_path = out_dir / "plan.json"
        plan_path.write_text(
            json.dumps({**plan, "aggregates": agg}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {summary_csv}")
        print(f"Wrote {agg_path}")
        print(
            f"aggregates: n_ok={agg['n_ok']}/{agg['n_repeats']} "
            f"spearman={agg.get('test_spearman_mean')}±{agg.get('test_spearman_std')}"
        )
    finally:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one PE-DB dataset/datasheet (optional edit filters) with "
            "nested Optuna: N-fold CV or holdout_3 × N seeds, X trials per repeat."
        )
    )
    parser.add_argument("--model", required=True, help="deepprime, oped, or pridict2")
    parser.add_argument("--n", type=int, required=True, help="Folds (cv) or random seeds (holdout_3)")
    parser.add_argument(
        "-x",
        "--n-trials",
        type=int,
        required=True,
        dest="n_trials",
        help="Optuna trials per fold/seed",
    )
    parser.add_argument(
        "--protocol",
        choices=["auto", "cv", "holdout_3"],
        default="auto",
        help="auto picks cv vs holdout_3 from row count (default: auto)",
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=DEFAULT_SIZE_THRESHOLD,
        help=f"Rows below this use CV when --protocol auto (default {DEFAULT_SIZE_THRESHOLD})",
    )
    parser.add_argument("--dataset-name", default=None, help="Label for this run (default: derived)")
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--cell-line", action="append", default=[])
    parser.add_argument("--pe-system", action="append", default=[])
    parser.add_argument(
        "--edit-type",
        action="append",
        default=[],
        help="sub, ins, and/or del (repeatable)",
    )
    parser.add_argument("--edit-length", action="append", type=int, default=[])
    parser.add_argument("--edit-scope", action="append", default=[])
    parser.add_argument("--experimental-method", action="append", default=[])
    parser.add_argument("--target-context", action="append", default=[])
    parser.add_argument("--scaffold-name", action="append", default=[])
    parser.add_argument("--edit-efficiency-min", type=float, default=None)
    parser.add_argument("--edit-efficiency-max", type=float, default=None)
    parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pool matching datasheets then split by shared locus (default on)",
    )
    parser.add_argument("--use-original-fold", action="store_true", default=False)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--train-pct", type=float, default=DEFAULT_HOLDOUT_TRAIN_PCT)
    parser.add_argument("--val-pct", type=float, default=DEFAULT_HOLDOUT_VAL_PCT)
    parser.add_argument("--test-pct", type=float, default=DEFAULT_HOLDOUT_TEST_PCT)
    parser.add_argument("--inner-train-pct", type=float, default=DEFAULT_INNER_TRAIN_PCT)
    parser.add_argument("--inner-val-pct", type=float, default=DEFAULT_INNER_VAL_PCT)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only this 0-based fold/seed (cluster fan-out)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/datasheet_benchmark/<run-id>)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Probe size and print the plan")
    parser.add_argument("--no-write-preset", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Register best weights per repeat but skip peen evaluate",
    )
    parser.add_argument(
        "--fixed-hyperparameters-json",
        default=None,
        help="JSON object merged into every trial (e.g. smoke epoch caps)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold/seed whose state file already exists under out-dir",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        print("Error: --n must be >= 1", file=sys.stderr)
        return 2
    if args.n_trials < 1:
        print("Error: --n-trials must be >= 1", file=sys.stderr)
        return 2
    if not args.study and not args.dataset:
        print("Error: provide --study and/or --dataset (a catalog dataset or datasheet)", file=sys.stderr)
        return 2

    _configure_logging()
    bootstrap_peen()

    from pe_ensemble.library import model_registry, supported_models

    model = args.model.strip().lower()
    if model not in set(supported_models()):
        print(f"Error: unknown model {model!r}; supported: {', '.join(supported_models())}", file=sys.stderr)
        return 2
    try:
        from pe_ensemble.training.search_spaces import get_search_space

        get_search_space(model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    filter_kwargs = filter_kwargs_from_args(args)
    label = dataset_label(args)
    print("======================================")
    print("Datasheet benchmark")
    print("======================================")
    print(f"model:     {model}")
    print(f"label:     {label}")
    print(f"n:         {args.n}")
    print(f"n_trials:  {args.n_trials}")
    print(f"device:    {args.device}")
    print(f"filters:   { {k: v for k, v in filter_kwargs.items() if v is not None} }")
    print()

    n_rows, probe = probe_n_rows(filter_kwargs, merge=args.merge)
    protocol, reason = choose_protocol(
        n_rows,
        size_threshold=args.size_threshold,
        override=args.protocol,
    )
    print(f"n_rows:    {n_rows}  (pedb filter --summary-only)")
    print(f"protocol:  {protocol}")
    print(f"reason:    {reason}")
    if probe.get("skipped"):
        print(f"skipped:   {probe['skipped']}")
    print()

    if n_rows == 0:
        print("Error: no matching edit rows. Check --study/--dataset/--edit-type.", file=sys.stderr)
        return 1

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(
        args.out_dir
        or (REPO_ROOT / "results" / "datasheet_benchmark" / run_id)
    )
    plan = {
        "run_id": run_id,
        "model": model,
        "dataset_name": label,
        "protocol": protocol,
        "protocol_reason": reason,
        "n": args.n,
        "n_trials": args.n_trials,
        "n_rows": n_rows,
        "size_threshold": args.size_threshold,
        "base_seed": args.base_seed,
        "device": args.device,
        "filters": {k: v for k, v in filter_kwargs.items() if v is not None},
        "merge": args.merge,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        print("dry-run: not fetching model-format data or running Optuna")
        return 0

    model_format = model_registry.get(model).pe_db_format
    print(f"Fetching {model_format} rows via pe_db.library (same path as pedb filter)...")
    frame = fetch_model_format_frame(
        model_format=model_format,
        filter_kwargs=filter_kwargs,
        merge=args.merge,
    )
    print(f"Loaded {len(frame)} model-format rows")

    repeats: list[tuple[str, int, pd.DataFrame]] = []
    if protocol == "cv":
        cv_frame = assign_cv_folds(
            frame,
            cv_folds=args.n,
            random_state=args.base_seed,
            use_original_fold=args.use_original_fold,
        )
        labels = fold_labels(cv_frame)
        print(f"CV folds: {labels}  counts={split_counts(cv_frame)}")
        seeds = repeat_seeds(args.base_seed, args.n)
        for index, fold_label in enumerate(labels):
            seed = seeds[index]
            assigned = remap_cv_fold_to_nested_holdout(
                cv_frame,
                fold_label,
                inner_random_state=seed,
                inner_train_pct=args.inner_train_pct,
                inner_val_pct=args.inner_val_pct,
            )
            repeats.append((fold_label, seed, assigned))
    else:
        seeds = repeat_seeds(args.base_seed, args.n)
        for seed in seeds:
            assigned = assign_holdout_3(
                frame,
                random_state=seed,
                train_pct=args.train_pct,
                val_pct=args.val_pct,
                test_pct=args.test_pct,
                use_original_fold=args.use_original_fold,
            )
            repeats.append((f"seed_{seed}", seed, assigned))

    if args.index is not None:
        if not 0 <= args.index < len(repeats):
            print(f"Error: --index {args.index} out of range 0..{len(repeats) - 1}", file=sys.stderr)
            return 2
        repeats = [repeats[args.index]]
        print(f"Restricted to --index {args.index}: {repeats[0][0]}")

    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    from pe_ensemble.training.dataset_key import dataset_preset_key

    base_preset = dataset_preset_key(
        study=filter_kwargs.get("study"),
        dataset=filter_kwargs.get("dataset"),
        cell_line=filter_kwargs.get("cell_line"),
        pe_system=filter_kwargs.get("pe_system"),
    ) or label.replace("-", "_")

    extra_hp: Optional[dict[str, Any]] = None
    if args.fixed_hyperparameters_json:
        extra_hp = json.loads(args.fixed_hyperparameters_json)
        if not isinstance(extra_hp, dict):
            print("Error: --fixed-hyperparameters-json must be a JSON object", file=sys.stderr)
            return 2

    for repeat_id, seed, assigned in repeats:
        state_path = state_dir / f"{repeat_id}.json"
        if args.skip_existing and state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            needs_eval = (
                not args.skip_eval
                and existing.get("weights_id")
                and existing.get("status") in {"tuned", None}
                and existing.get("eval") is None
            )
            if needs_eval:
                print(f"Resume eval {repeat_id} (weights={existing.get('weights_id')})")
                eval_block = evaluate_assigned(
                    model=model,
                    dataset_name=label,
                    repeat_id=repeat_id,
                    weights_id=str(existing["weights_id"]),
                    assigned=assigned,
                    device=args.device,
                )
                existing.update(eval_block)
                state_path.write_text(json.dumps(existing, default=str), encoding="utf-8")
            else:
                print(f"SKIP existing {repeat_id} ({state_path})")
            rows.append(existing)
            continue
        print("======================================")
        print(f"{protocol} repeat {repeat_id} (seed={seed})")
        print("======================================")
        result = run_tune_and_eval(
            model=model,
            dataset_name=label,
            filter_kwargs=filter_kwargs,
            assigned=assigned,
            n_trials=args.n_trials,
            device=args.device,
            seed=seed,
            repeat_id=repeat_id,
            study_name=f"datasheet-bench__{model}__{label}__{repeat_id}",
            dataset_preset_key=f"{base_preset}/{repeat_id}",
            notes=f"datasheet-benchmark {protocol} {repeat_id} seed={seed}",
            no_write_preset=args.no_write_preset,
            extra_hyperparameters=extra_hp,
            skip_eval=True,
        )
        record = {**plan, **result}
        state_path.write_text(json.dumps(record, default=str), encoding="utf-8")
        if not args.skip_eval:
            eval_block = evaluate_assigned(
                model=model,
                dataset_name=label,
                repeat_id=repeat_id,
                weights_id=str(result["weights_id"]),
                assigned=assigned,
                device=args.device,
            )
            record.update(eval_block)
            state_path.write_text(json.dumps(record, default=str), encoding="utf-8")
        rows.append(record)

    write_outputs(
        out_dir=out_dir,
        run_id=run_id,
        plan=plan,
        rows=rows,
        state_dir=state_dir,
    )
    latest = REPO_ROOT / "results" / "datasheet_benchmark" / "LATEST_RUN_ID"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(run_id + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

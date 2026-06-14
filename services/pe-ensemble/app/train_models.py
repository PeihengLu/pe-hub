#!/usr/bin/env python3
"""CLI for running pe-ensemble model training (local dev or SLURM batch jobs)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from pe_common.devices import format_devices_for_cli

from .training.config import jobs_root
from .training.model_architecture import architecture_from_cli_args, merge_training_hyperparameters
from .compute.device_scheduler import get_scheduler
from .training.jobs import create_job, get_job, read_logs, wait_for_job
from .training.runner import TrainingError
from .training.schemas import SplitQueryParams, TrainingRequest


def _parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _build_request(args: argparse.Namespace) -> TrainingRequest:
    split = SplitQueryParams(
        split_strategy=args.split_strategy,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        cv_folds=args.cv_folds,
        use_original_fold=args.use_original_fold,
        original_fold_test_value=args.original_fold_test_value,
        split_random_state=args.split_random_state,
        merge=args.merge,
    )
    base = _parse_json_object(args.hyperparameters_json) or {}
    if args.pretrained_weights:
        base["load_pretrained"] = True
        base["weights"] = args.pretrained_weights
    hyperparameters = merge_training_hyperparameters(
        args.model,
        base,
        architecture_from_cli_args(args.model, args),
    )
    return TrainingRequest(
        model_name=args.model,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        hyperparameters=hyperparameters or None,
        split=split,
        study=args.study or None,
        dataset=args.dataset or None,
        cell_line=args.cell_line or None,
        pe_system=args.pe_system or None,
        edit_type=args.edit_type or None,
        edit_length=args.edit_length or None,
        edit_scope=args.edit_scope or None,
        experimental_method=args.experimental_method or None,
        target_context=args.target_context or None,
        scaffold_name=args.scaffold_name or None,
        model_kwargs=_parse_json_object(args.model_kwargs_json),
        notes=args.notes,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pe-ensemble model on PE-DB data or inline records.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["deepprime", "oped", "pridict2"],
        help="Model to train",
    )
    parser.add_argument(
        "--dataset-source",
        default="pe-db",
        help="Provenance label stored in the weight manifest (default: pe-db)",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset label stored in the manifest and used in job listings",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="PE-DB dataset filter (repeatable)",
    )
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--cell-line", action="append", default=[])
    parser.add_argument("--pe-system", action="append", default=[])
    parser.add_argument("--edit-type", action="append", default=[])
    parser.add_argument("--edit-length", action="append", type=int, default=[])
    parser.add_argument("--edit-scope", action="append", default=[])
    parser.add_argument("--experimental-method", action="append", default=[])
    parser.add_argument("--target-context", action="append", default=[])
    parser.add_argument("--scaffold-name", action="append", default=[])
    parser.add_argument("--split-strategy", default="holdout_3", choices=["none", "holdout_2", "holdout_3", "cv"])
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--val-pct", type=float, default=0.15)
    parser.add_argument("--test-pct", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--use-original-fold", action="store_true")
    parser.add_argument(
        "--original-fold-test-value",
        type=float,
        default=-1.0,
        help="original_fold value treated as test when --use-original-fold is set",
    )
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--hyperparameters-json", default=None, help="Training hyperparameters as JSON object")
    parser.add_argument(
        "--pretrained-weights",
        default=None,
        help="Registered weight set ID to fine-tune from (sets load_pretrained=true)",
    )
    parser.add_argument("--model-kwargs-json", default=None, help="Model constructor kwargs as JSON object")

    arch = parser.add_argument_group("model architecture (optional; merged into hyperparameters)")
    arch.add_argument(
        "--dp-hidden-size",
        type=int,
        default=None,
        help="DeepPrime GRU hidden size when training from scratch (ignored with load_pretrained)",
    )
    arch.add_argument(
        "--dp-num-layers",
        type=int,
        default=None,
        help="DeepPrime GRU layer count when training from scratch",
    )
    arch.add_argument("--oped-embedding-size", type=int, default=None, help="OPED token embedding dimension")
    arch.add_argument(
        "--oped-ffn-dim",
        type=int,
        default=None,
        help="OPED transformer feed-forward dim (applied to Target/PBS/RT branches)",
    )
    arch.add_argument(
        "--oped-encoder-layers",
        type=int,
        default=None,
        help="OPED transformer encoder depth per branch",
    )
    arch.add_argument("--oped-nhead", type=int, default=None, help="OPED multi-head attention head count")
    arch.add_argument("--oped-dropout", type=float, default=None, help="OPED dropout rate")
    arch.add_argument("--pridict2-embed-dim", type=int, default=None, help="PRIDICT2 sequence embedding dimension")
    arch.add_argument("--pridict2-z-dim", type=int, default=None, help="PRIDICT2 latent dimension")
    arch.add_argument(
        "--pridict2-num-hidden-layers",
        type=int,
        default=None,
        help="PRIDICT2 RNN hidden layer count",
    )
    arch.add_argument("--pridict2-dropout", type=float, default=None, help="PRIDICT2 dropout rate")

    parser.add_argument("--notes", default=None)
    parser.add_argument("--pe-db-url", default=None, help="Override PE_DB_URL for this run")
    parser.add_argument("--weights-root", default=None, help="Override WEIGHTS_ROOT for this run")
    parser.add_argument("--jobs-root", default=None, help="Override TRAINING_JOBS_ROOT for this run")
    parser.add_argument(
        "--job-id",
        default=None,
        help="Use a fixed job id (created if missing). Useful for SLURM array jobs.",
    )
    parser.add_argument(
        "--run-existing-job",
        action="store_true",
        help="Execute a previously queued job by --job-id instead of building a new request",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="Create the job directory and exit without training (for sbatch wrappers)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device id (e.g. cuda:0, mps, cpu). Default: auto (best accelerator)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available compute devices and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        print(format_devices_for_cli())
        return 0

    if args.pe_db_url:
        os.environ["PE_DB_URL"] = args.pe_db_url
    if args.weights_root:
        os.environ["WEIGHTS_ROOT"] = args.weights_root
    if args.jobs_root:
        os.environ["TRAINING_JOBS_ROOT"] = args.jobs_root

    try:
        if args.run_existing_job:
            if not args.job_id:
                parser.error("--run-existing-job requires --job-id")
            manifest = get_job(args.job_id)
            if manifest["status"] not in ("queued", "failed"):
                raise TrainingError(f"Job {args.job_id} is already {manifest['status']}")
            request_path = jobs_root() / args.job_id / "request.json"
            with open(request_path, encoding="utf-8") as handle:
                request = TrainingRequest.model_validate(json.load(handle))
            get_scheduler().submit_training(args.job_id, request)
            manifest = wait_for_job(args.job_id)
            if manifest.get("result") is None:
                raise TrainingError(manifest.get("error") or "Training failed")
            print(json.dumps(manifest["result"], indent=2))
            return 0

        request = _build_request(args)
        job_id = args.job_id
        if job_id:
            try:
                get_job(job_id)
            except FileNotFoundError:
                job_id = create_job(request, job_id=job_id)
        else:
            job_id = create_job(request)

        print(f"job_id={job_id}")
        print(f"jobs_root={jobs_root()}")

        if args.queue_only:
            print("Job queued.")
            return 0

        scheduler = get_scheduler()
        scheduler.submit_training(job_id, request)
        manifest = wait_for_job(job_id)
        if manifest.get("result") is None:
            raise TrainingError(manifest.get("error") or "Training failed")
        print(json.dumps(manifest["result"], indent=2))
        return 0
    except TrainingError as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        if args.job_id:
            log, _ = read_logs(args.job_id)
            if log:
                print(log, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

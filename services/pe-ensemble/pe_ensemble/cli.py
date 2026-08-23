#!/usr/bin/env python3
"""Command-line interface for PE Ensemble (train, tune, evaluate, ensemble)."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from pe_common.devices import format_devices_for_cli

from pe_ensemble._bootstrap import ensure_service_root_on_path

ensure_service_root_on_path()

from app.compute.device_scheduler import get_scheduler  # noqa: E402
from app.ensemble.combine import COMBINE_METHODS  # noqa: E402
from app.ensemble.jobs import (  # noqa: E402
    create_job as create_ensemble_job,
    get_job as get_ensemble_job,
    list_jobs as list_ensemble_jobs,
    read_logs as read_ensemble_logs,
    wait_for_job as wait_for_ensemble_job,
)
from app.evaluation.jobs import (  # noqa: E402
    create_job as create_eval_job,
    get_job as get_eval_job,
    list_jobs as list_eval_jobs,
    read_logs as read_eval_logs,
    wait_for_job as wait_for_eval_job,
)
from app.models.registry import model_registry  # noqa: E402
from app.training.config import jobs_root, supported_models  # noqa: E402
from app.training.jobs import (  # noqa: E402
    create_job as create_train_job,
    get_job as get_train_job,
    list_jobs as list_train_jobs,
    read_logs as read_train_logs,
    wait_for_job as wait_for_train_job,
)
from app.training.runner import TrainingError  # noqa: E402
from app.training.schemas import TrainingRequest  # noqa: E402
from app.training.tune_jobs import (  # noqa: E402
    create_job as create_tune_job,
    get_job as get_tune_job,
    list_jobs as list_tune_jobs,
    read_logs as read_tune_logs,
    wait_for_job as wait_for_tune_job,
)
from app.training.tune_study import execute_tuning  # noqa: E402
from pe_ensemble.cli_common import (  # noqa: E402
    add_architecture_flags,
    add_env_flags,
    add_filter_flags,
    add_split_flags,
    apply_env_overrides,
    build_ensemble_request,
    build_evaluation_request,
    build_training_request,
    build_tuning_request,
    prepare_runtime,
)
from pe_ensemble.library import execute_evaluation, execute_ensemble, execute_training  # noqa: E402


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="Train a model on PE-DB data")
    parser.add_argument("--model", required=True, choices=list(supported_models()))
    parser.add_argument("--dataset-source", default="pe-db")
    parser.add_argument("--dataset-name", required=True)
    add_filter_flags(parser)
    add_split_flags(parser)
    parser.add_argument("--hyperparameters-json", default=None)
    parser.add_argument(
        "--hyperparameter-mode",
        default="merge",
        choices=["merge", "replace"],
    )
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--model-kwargs-json", default=None)
    add_architecture_flags(parser)
    parser.add_argument("--notes", default=None)
    add_env_flags(parser)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--run-existing-job", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.set_defaults(func=cmd_train)


def _add_tune_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("tune", help="Optuna hyperparameter tuning")
    parser.add_argument("--model", required=True, choices=list(supported_models()))
    parser.add_argument("--dataset-source", default="pe-db")
    parser.add_argument("--dataset-name", required=True)
    add_filter_flags(parser)
    add_split_flags(parser)
    parser.add_argument("--fixed-hyperparameters-json", default=None)
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--notes", default=None)
    add_env_flags(parser)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--study-storage", default=None)
    parser.add_argument("--write-preset", default=None)
    parser.add_argument("--no-write-preset", action="store_true")
    parser.add_argument("--register-best-weights", action="store_true")
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Submit to the device scheduler instead of running in-process",
    )
    parser.add_argument("--job-id", default=None)
    parser.set_defaults(func=cmd_tune)


def _add_evaluate_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("evaluate", help="Benchmark a registered weight set")
    parser.add_argument("--model", required=True, choices=list(supported_models()))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--benchmark-name", default=None)
    add_filter_flags(parser)
    add_split_flags(parser)
    parser.set_defaults(
        split_strategy="holdout_2",
        train_pct=0.8,
        val_pct=None,
        test_pct=0.2,
        use_original_fold=True,
    )
    parser.add_argument(
        "--custom-benchmark",
        action="store_true",
        help="Use CLI filters instead of the weight set training metadata",
    )
    parser.add_argument("--allow-data-leak", action="store_true")
    add_env_flags(parser)
    parser.add_argument("--sync", action="store_true", help="Run in-process without job queue")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--queue-only", action="store_true")
    parser.set_defaults(func=cmd_evaluate)


def _add_ensemble_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("ensemble", help="Fuse member model predictions")
    parser.add_argument("--ensemble-name", required=True)
    parser.add_argument("--combine", default="mean", choices=list(COMBINE_METHODS))
    parser.add_argument("--combine-options-json", default=None)
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        help="Member as model:weights or model:weights:member_weight (repeatable)",
    )
    add_filter_flags(parser)
    add_split_flags(parser)
    parser.set_defaults(
        split_strategy="holdout_2",
        train_pct=0.8,
        val_pct=None,
        test_pct=0.2,
        use_original_fold=True,
    )
    add_env_flags(parser)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--queue-only", action="store_true")
    parser.set_defaults(func=cmd_ensemble)


def _add_list_parsers(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("methods", help="List ensemble combine methods").set_defaults(func=cmd_methods)
    models_p = sub.add_parser("models", help="List supported models")
    models_p.set_defaults(func=cmd_models)
    weights_p = sub.add_parser("weights", help="List registered weights for a model")
    weights_p.add_argument("--model", required=True, choices=list(supported_models()))
    weights_p.set_defaults(func=cmd_weights)
    sub.add_parser("devices", help="List compute devices").set_defaults(func=cmd_devices)


def _add_jobs_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("jobs", help="List recent jobs")
    parser.add_argument(
        "--kind",
        default="train",
        choices=["train", "tune", "evaluate", "ensemble"],
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.set_defaults(func=cmd_jobs)


def _add_logs_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("logs", help="Read job logs")
    parser.add_argument("--kind", default="train", choices=["train", "tune", "evaluate", "ensemble"])
    parser.add_argument("job_id")
    parser.add_argument("--offset", type=int, default=0)
    parser.set_defaults(func=cmd_logs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pe-ensemble",
        description="PE Ensemble headless tools (train, tune, evaluate, ensemble).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(sub)
    _add_tune_parser(sub)
    _add_evaluate_parser(sub)
    _add_ensemble_parser(sub)
    _add_list_parsers(sub)
    _add_jobs_parser(sub)
    _add_logs_parser(sub)
    return parser


def _run_queued_job(
    *,
    kind: str,
    job_id: str,
    submit,
    wait,
    queue_only: bool,
) -> int:
    print(f"job_id={job_id}")
    if queue_only:
        print("Job queued.")
        return 0
    submit(job_id)
    manifest = wait(job_id)
    if manifest.get("result") is None and manifest.get("status") not in ("skipped",):
        raise TrainingError(manifest.get("error") or f"{kind} job failed")
    print(json.dumps(manifest.get("result") or manifest, indent=2, default=str))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    apply_env_overrides(args)
    if args.run_existing_job:
        if not args.job_id:
            raise TrainingError("--run-existing-job requires --job-id")
        manifest = get_train_job(args.job_id)
        if manifest["status"] not in ("queued", "failed"):
            raise TrainingError(f"Job {args.job_id} is already {manifest['status']}")
        request_path = jobs_root() / args.job_id / "request.json"
        with open(request_path, encoding="utf-8") as handle:
            request = TrainingRequest.model_validate(json.load(handle))
        get_scheduler().submit_training(args.job_id, request)
        manifest = wait_for_train_job(args.job_id)
        if manifest.get("result") is None:
            raise TrainingError(manifest.get("error") or "Training failed")
        print(json.dumps(manifest["result"], indent=2))
        return 0

    request = build_training_request(args)
    job_id = args.job_id
    if job_id:
        try:
            get_train_job(job_id)
        except FileNotFoundError:
            job_id = create_train_job(request, job_id=job_id)
    else:
        job_id = create_train_job(request)

    print(f"job_id={job_id}")
    print(f"jobs_root={jobs_root()}")
    if args.queue_only:
        print("Job queued.")
        return 0

    get_scheduler().submit_training(job_id, request)
    manifest = wait_for_train_job(job_id)
    if manifest.get("result") is None:
        raise TrainingError(manifest.get("error") or "Training failed")
    print(json.dumps(manifest["result"], indent=2))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    apply_env_overrides(args)
    request = build_tuning_request(args)

    if args.queue:
        job_id = args.job_id
        if job_id:
            try:
                get_tune_job(job_id)
            except FileNotFoundError:
                job_id = create_tune_job(request, job_id=job_id)
        else:
            job_id = create_tune_job(request)
        return _run_queued_job(
            kind="tune",
            job_id=job_id,
            submit=lambda jid: get_scheduler().submit_tuning(jid, request),
            wait=wait_for_tune_job,
            queue_only=False,
        )

    summary = execute_tuning(request, device_id=request.training.device)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    apply_env_overrides(args)
    request = build_evaluation_request(args)

    if args.sync:
        result = execute_evaluation(request, device_id=request.device)
        print(json.dumps(result, indent=2, default=str))
        return 0

    job_id = args.job_id
    if job_id:
        try:
            get_eval_job(job_id)
        except FileNotFoundError:
            job_id = create_eval_job(request, job_id=job_id)
    else:
        job_id = create_eval_job(request)

    return _run_queued_job(
        kind="evaluate",
        job_id=job_id,
        submit=lambda jid: get_scheduler().submit_evaluation(jid, request),
        wait=wait_for_eval_job,
        queue_only=args.queue_only,
    )


def cmd_ensemble(args: argparse.Namespace) -> int:
    apply_env_overrides(args)
    if len(args.member) < 2:
        raise TrainingError("ensemble requires at least two --member entries")
    request = build_ensemble_request(args)

    if args.sync:
        result = execute_ensemble(request, device_id=request.device)
        print(json.dumps(result, indent=2, default=str))
        return 0

    job_id = args.job_id
    if job_id:
        try:
            get_ensemble_job(job_id)
        except FileNotFoundError:
            job_id = create_ensemble_job(request, job_id=job_id)
    else:
        job_id = create_ensemble_job(request)

    return _run_queued_job(
        kind="ensemble",
        job_id=job_id,
        submit=lambda jid: get_scheduler().submit_ensemble(jid, request),
        wait=wait_for_ensemble_job,
        queue_only=args.queue_only,
    )


def cmd_methods(args: argparse.Namespace) -> int:
    del args
    from pe_ensemble.library import combine_method_help

    for entry in combine_method_help():
        print(f"{entry['method']}: {entry['description']}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(model_registry.list_catalog_entries(), indent=2))
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    entries = model_registry.list_weight_entries(args.model)
    print(json.dumps(entries, indent=2, default=str))
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    del args
    print(format_devices_for_cli())
    snapshot = get_scheduler().device_snapshot()
    print(json.dumps(snapshot, indent=2))
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    if args.kind == "train":
        jobs = list_train_jobs(limit=args.limit)
    elif args.kind == "tune":
        jobs = list_tune_jobs(limit=args.limit)
    elif args.kind == "evaluate":
        jobs = list_eval_jobs(limit=args.limit)
    else:
        jobs = list_ensemble_jobs(limit=args.limit)
    print(json.dumps(jobs, indent=2, default=str))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    readers = {
        "train": (get_train_job, read_train_logs),
        "tune": (get_tune_job, read_tune_logs),
        "evaluate": (get_eval_job, read_eval_logs),
        "ensemble": (get_ensemble_job, read_ensemble_logs),
    }
    get_job, read_logs = readers[args.kind]
    manifest = get_job(args.job_id)
    chunk, next_offset = read_logs(args.job_id, offset=args.offset)
    payload = {
        "job_id": args.job_id,
        "status": manifest.get("status"),
        "offset": args.offset,
        "next_offset": next_offset,
        "log": chunk,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    early_exit = prepare_runtime(argv)
    if early_exit is not None:
        return early_exit

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TrainingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command-line interface for PE Ensemble (train, tune, evaluate, ensemble)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
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
from app.ensemble.schemas import EnsembleMember, EnsembleRequest  # noqa: E402
from app.evaluation.jobs import (  # noqa: E402
    create_job as create_eval_job,
    get_job as get_eval_job,
    list_jobs as list_eval_jobs,
    read_logs as read_eval_logs,
    wait_for_job as wait_for_eval_job,
)
from app.evaluation.schemas import EvaluationRequest  # noqa: E402
from app.models.registry import model_registry  # noqa: E402
from app.training.config import enable_cli_pe_db_access, jobs_root, supported_models  # noqa: E402
from app.training.jobs import (  # noqa: E402
    create_job as create_train_job,
    get_job as get_train_job,
    list_jobs as list_train_jobs,
    read_logs as read_train_logs,
    wait_for_job as wait_for_train_job,
)
from app.training.model_architecture import (  # noqa: E402
    architecture_from_cli_args,
    merge_training_hyperparameters,
)
from app.training.pe_db_access import PeDbAccessError, reload_pe_db_plugins  # noqa: E402
from app.training.runner import TrainingError  # noqa: E402
from app.training.schemas import SplitQueryParams, TrainingRequest  # noqa: E402
from app.training.tune_jobs import (  # noqa: E402
    create_job as create_tune_job,
    get_job as get_tune_job,
    list_jobs as list_tune_jobs,
    read_logs as read_tune_logs,
    wait_for_job as wait_for_tune_job,
)
from app.training.tune_study import execute_tuning  # noqa: E402
from app.training.tuning_schemas import TuningRequest  # noqa: E402
from pe_ensemble.library import (  # noqa: E402
    combine_method_help,
    execute_evaluation,
    execute_ensemble,
)


def _early_parse(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plugins-root")
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_known_args(argv)[0]


def _bootstrap_plugins() -> List[str]:
    from app.plugin_loader import load_active_plugins

    return load_active_plugins()


def _sync_pe_db_plugins() -> None:
    try:
        reload_pe_db_plugins()
    except PeDbAccessError:
        pass


def _parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _optional_list(values: List[Any]) -> Optional[List[Any]]:
    return values or None


def _add_split_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-strategy",
        default="holdout_3",
        choices=["none", "holdout_2", "holdout_3", "cv"],
    )
    # Defaults applied in _build_split by strategy (CV must not inherit holdout pcts).
    parser.add_argument("--train-pct", type=float, default=None)
    parser.add_argument("--val-pct", type=float, default=None)
    parser.add_argument("--test-pct", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument(
        "--use-original-fold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use author original_fold where available (--no-use-original-fold to force random)",
    )
    parser.add_argument(
        "--original-fold-test-value",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--merge", action="store_true")


def _add_filter_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--cell-line", action="append", default=[])
    parser.add_argument("--pe-system", action="append", default=[])
    parser.add_argument("--edit-type", action="append", default=[])
    parser.add_argument("--edit-length", action="append", type=int, default=[])
    parser.add_argument("--edit-scope", action="append", default=[])
    parser.add_argument("--experimental-method", action="append", default=[])
    parser.add_argument("--target-context", action="append", default=[])
    parser.add_argument("--scaffold-name", action="append", default=[])
    parser.add_argument("--edit-efficiency-min", type=float, default=None)
    parser.add_argument("--edit-efficiency-max", type=float, default=None)


def _add_architecture_flags(parser: argparse.ArgumentParser) -> None:
    arch = parser.add_argument_group("model architecture")
    arch.add_argument("--dp-hidden-size", type=int, default=None)
    arch.add_argument("--dp-num-layers", type=int, default=None)
    arch.add_argument("--oped-embedding-size", type=int, default=None)
    arch.add_argument("--oped-ffn-dim", type=int, default=None)
    arch.add_argument("--oped-encoder-layers", type=int, default=None)
    arch.add_argument("--oped-nhead", type=int, default=None)
    arch.add_argument("--oped-dropout", type=float, default=None)
    arch.add_argument("--pridict2-embed-dim", type=int, default=None)
    arch.add_argument("--pridict2-z-dim", type=int, default=None)
    arch.add_argument("--pridict2-num-hidden-layers", type=int, default=None)
    arch.add_argument("--pridict2-dropout", type=float, default=None)


def _add_env_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--plugins-root", default=None)
    parser.add_argument("--jobs-root", default=None)
    parser.add_argument("--presets-root", default=None)
    parser.add_argument("--tuning-jobs-root", default=None)
    parser.add_argument("--device", default="auto")


def _apply_env_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "plugins_root", None):
        os.environ["PLUGINS_ROOT"] = args.plugins_root
    if getattr(args, "weights_root", None):
        os.environ["WEIGHTS_ROOT"] = args.weights_root
    if getattr(args, "jobs_root", None):
        os.environ["TRAINING_JOBS_ROOT"] = args.jobs_root
    if getattr(args, "presets_root", None):
        os.environ["TRAINING_PRESETS_ROOT"] = args.presets_root
    if getattr(args, "tuning_jobs_root", None):
        os.environ["TUNING_JOBS_ROOT"] = args.tuning_jobs_root


def _build_split(args: argparse.Namespace) -> SplitQueryParams:
    strategy = args.split_strategy
    train_pct = args.train_pct
    val_pct = args.val_pct
    test_pct = args.test_pct

    if strategy == "holdout_3":
        train_pct = 0.7 if train_pct is None else train_pct
        val_pct = 0.15 if val_pct is None else val_pct
        test_pct = 0.15 if test_pct is None else test_pct
    elif strategy == "holdout_2":
        train_pct = 0.8 if train_pct is None else train_pct
        val_pct = None
        test_pct = 0.2 if test_pct is None else test_pct
    elif strategy == "cv":
        # Pure k-fold unless the caller passes --test-pct for an outer holdout.
        train_pct = None
        val_pct = None
    elif strategy == "none":
        train_pct = None
        val_pct = None
        test_pct = None

    return SplitQueryParams(
        split_strategy=strategy,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        cv_folds=args.cv_folds,
        use_original_fold=args.use_original_fold,
        original_fold_test_value=args.original_fold_test_value,
        split_random_state=args.split_random_state,
        merge=args.merge,
    )


def build_training_request(args: argparse.Namespace) -> TrainingRequest:
    base = _parse_json_object(getattr(args, "hyperparameters_json", None)) or {}
    if getattr(args, "pretrained_weights", None):
        base["load_pretrained"] = True
        base["weights"] = args.pretrained_weights
    hyperparameters = merge_training_hyperparameters(
        args.model,
        base,
        architecture_from_cli_args(args.model, args),
    )
    return TrainingRequest(
        model_name=args.model,
        dataset_source=getattr(args, "dataset_source", "pe-db"),
        dataset_name=args.dataset_name,
        hyperparameters=hyperparameters or None,
        hyperparameter_mode=getattr(args, "hyperparameter_mode", "merge"),
        split=_build_split(args),
        study=_optional_list(args.study),
        dataset=_optional_list(args.dataset),
        cell_line=_optional_list(args.cell_line),
        pe_system=_optional_list(args.pe_system),
        edit_type=_optional_list(args.edit_type),
        edit_length=_optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=_optional_list(args.edit_scope),
        experimental_method=_optional_list(args.experimental_method),
        target_context=_optional_list(args.target_context),
        scaffold_name=_optional_list(args.scaffold_name),
        model_kwargs=_parse_json_object(getattr(args, "model_kwargs_json", None)),
        notes=getattr(args, "notes", None),
        device=args.device,
    )


def build_tuning_request(args: argparse.Namespace) -> TuningRequest:
    fixed = _parse_json_object(getattr(args, "fixed_hyperparameters_json", None)) or {}
    training = TrainingRequest(
        model_name=args.model,
        dataset_source=getattr(args, "dataset_source", "pe-db"),
        dataset_name=args.dataset_name,
        hyperparameters=fixed or None,
        hyperparameter_mode="replace",
        split=_build_split(args),
        study=_optional_list(args.study),
        dataset=_optional_list(args.dataset),
        cell_line=_optional_list(args.cell_line),
        pe_system=_optional_list(args.pe_system),
        edit_type=_optional_list(args.edit_type),
        edit_length=_optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=_optional_list(args.edit_scope),
        experimental_method=_optional_list(args.experimental_method),
        target_context=_optional_list(args.target_context),
        scaffold_name=_optional_list(args.scaffold_name),
        model_kwargs=_parse_json_object(getattr(args, "model_kwargs_json", None)),
        notes=getattr(args, "notes", None),
        device=args.device,
    )
    return TuningRequest(
        training=training,
        n_trials=args.n_trials,
        study_name=args.study_name,
        study_storage=args.study_storage,
        write_preset=args.write_preset,
        no_write_preset=args.no_write_preset,
        register_best_weights=args.register_best_weights,
    )


def build_evaluation_request(args: argparse.Namespace) -> EvaluationRequest:
    return EvaluationRequest(
        model_name=args.model,
        benchmark_name=args.benchmark_name,
        weights=args.weights,
        split=_build_split(args),
        study=_optional_list(args.study),
        dataset=_optional_list(args.dataset),
        cell_line=_optional_list(args.cell_line),
        pe_system=_optional_list(args.pe_system),
        edit_type=_optional_list(args.edit_type),
        edit_length=_optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=_optional_list(args.edit_scope),
        experimental_method=_optional_list(args.experimental_method),
        target_context=_optional_list(args.target_context),
        scaffold_name=_optional_list(args.scaffold_name),
        device=args.device,
        auto_training_benchmark=not args.custom_benchmark,
        allow_data_leak=args.allow_data_leak,
    )


def parse_ensemble_member(raw: str) -> EnsembleMember:
    parts = raw.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid --member {raw!r}; expected model:weights or model:weights:member_weight"
        )
    model_name, weights = parts[0], parts[1]
    member_weight = float(parts[2]) if len(parts) > 2 else None
    return EnsembleMember(
        model_name=model_name,
        weights=weights,
        member_weight=member_weight,
    )


def build_ensemble_request(args: argparse.Namespace) -> EnsembleRequest:
    combine_options = _parse_json_object(args.combine_options_json) or {}
    return EnsembleRequest(
        ensemble_name=args.ensemble_name,
        combine=args.combine,
        combine_options=combine_options,
        members=[parse_ensemble_member(raw) for raw in args.member],
        split=_build_split(args),
        study=_optional_list(args.study),
        dataset=_optional_list(args.dataset),
        cell_line=_optional_list(args.cell_line),
        pe_system=_optional_list(args.pe_system),
        edit_type=_optional_list(args.edit_type),
        edit_length=_optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=_optional_list(args.edit_scope),
        experimental_method=_optional_list(args.experimental_method),
        target_context=_optional_list(args.target_context),
        scaffold_name=_optional_list(args.scaffold_name),
        device=args.device,
    )


def _prepare_runtime(argv: Optional[List[str]]) -> Optional[int]:
    """Bootstrap plugins and handle global early-exit flags."""
    enable_cli_pe_db_access()
    early = _early_parse(argv)
    if early.list_devices:
        print(format_devices_for_cli())
        return 0
    if early.plugins_root:
        os.environ["PLUGINS_ROOT"] = early.plugins_root
    loaded = _bootstrap_plugins()
    if loaded:
        print(f"Loaded plugins: {', '.join(loaded)}", file=sys.stderr)
    _sync_pe_db_plugins()
    return None


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="Train a model on PE-DB data")
    parser.add_argument("--model", required=True, choices=list(supported_models()))
    parser.add_argument("--dataset-source", default="pe-db")
    parser.add_argument("--dataset-name", required=True)
    _add_filter_flags(parser)
    _add_split_flags(parser)
    parser.add_argument("--hyperparameters-json", default=None)
    parser.add_argument(
        "--hyperparameter-mode",
        default="merge",
        choices=["merge", "replace"],
    )
    parser.add_argument("--pretrained-weights", default=None)
    parser.add_argument("--model-kwargs-json", default=None)
    _add_architecture_flags(parser)
    parser.add_argument("--notes", default=None)
    _add_env_flags(parser)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--run-existing-job", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.set_defaults(func=cmd_train)


def _add_tune_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("tune", help="Optuna hyperparameter tuning")
    parser.add_argument("--model", required=True, choices=list(supported_models()))
    parser.add_argument("--dataset-source", default="pe-db")
    parser.add_argument("--dataset-name", required=True)
    _add_filter_flags(parser)
    _add_split_flags(parser)
    parser.add_argument("--fixed-hyperparameters-json", default=None)
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--notes", default=None)
    _add_env_flags(parser)
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
    _add_filter_flags(parser)
    _add_split_flags(parser)
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
    _add_env_flags(parser)
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
    _add_filter_flags(parser)
    _add_split_flags(parser)
    parser.set_defaults(
        split_strategy="holdout_2",
        train_pct=0.8,
        val_pct=None,
        test_pct=0.2,
        use_original_fold=True,
    )
    _add_env_flags(parser)
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
        prog=Path(sys.argv[0]).name if sys.argv else "peen",
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
    _apply_env_overrides(args)
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
    _apply_env_overrides(args)
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
    _apply_env_overrides(args)
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
    _apply_env_overrides(args)
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
    # Build the parser and handle shell completion before plugin/DB bootstrap
    # so tab-complete stays fast and does not touch PE-DB.
    parser = build_parser()
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    early_exit = _prepare_runtime(argv)
    if early_exit is not None:
        return early_exit

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

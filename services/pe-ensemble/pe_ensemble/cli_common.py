"""Shared helpers for the pe-ensemble CLI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from pe_common.devices import format_devices_for_cli

from pe_ensemble._bootstrap import ensure_service_root_on_path

ensure_service_root_on_path()

from app.ensemble.schemas import EnsembleMember, EnsembleRequest  # noqa: E402
from app.evaluation.schemas import EvaluationRequest  # noqa: E402
from app.training.model_architecture import (  # noqa: E402
    architecture_from_cli_args,
    merge_training_hyperparameters,
)
from app.training.pe_db_access import PeDbAccessError, reload_pe_db_plugins  # noqa: E402
from app.training.schemas import SplitQueryParams, TrainingRequest  # noqa: E402
from app.training.tuning_schemas import TuningRequest  # noqa: E402


def apply_default_pe_db_mode() -> None:
    if "PE_DB_MODE" not in os.environ:
        os.environ["PE_DB_MODE"] = "library"


def early_parse(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plugins-root")
    parser.add_argument("--pe-db-mode", choices=["http", "library"])
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_known_args(argv)[0]


def bootstrap_plugins() -> List[str]:
    from app.plugin_loader import load_active_plugins

    return load_active_plugins()


def sync_pe_db_plugins() -> None:
    try:
        reload_pe_db_plugins()
    except PeDbAccessError:
        pass


def parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def optional_list(values: List[Any]) -> Optional[List[Any]]:
    return values or None


def add_split_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-strategy",
        default="holdout_3",
        choices=["none", "holdout_2", "holdout_3", "cv"],
    )
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--val-pct", type=float, default=0.15)
    parser.add_argument("--test-pct", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--use-original-fold", action="store_true")
    parser.add_argument(
        "--original-fold-test-value",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--merge", action="store_true")


def add_filter_flags(parser: argparse.ArgumentParser) -> None:
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


def add_architecture_flags(parser: argparse.ArgumentParser) -> None:
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


def add_env_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pe-db-url", default=None)
    parser.add_argument(
        "--pe-db-mode",
        default=None,
        choices=["http", "library"],
        help="PE-DB transport (default for CLI: library)",
    )
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--plugins-root", default=None)
    parser.add_argument("--jobs-root", default=None)
    parser.add_argument("--presets-root", default=None)
    parser.add_argument("--tuning-jobs-root", default=None)
    parser.add_argument("--device", default="auto")


def apply_env_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "plugins_root", None):
        os.environ["PLUGINS_ROOT"] = args.plugins_root
    if getattr(args, "pe_db_url", None):
        os.environ["PE_DB_URL"] = args.pe_db_url
    if getattr(args, "pe_db_mode", None):
        os.environ["PE_DB_MODE"] = args.pe_db_mode
    if getattr(args, "weights_root", None):
        os.environ["WEIGHTS_ROOT"] = args.weights_root
    if getattr(args, "jobs_root", None):
        os.environ["TRAINING_JOBS_ROOT"] = args.jobs_root
    if getattr(args, "presets_root", None):
        os.environ["TRAINING_PRESETS_ROOT"] = args.presets_root
    if getattr(args, "tuning_jobs_root", None):
        os.environ["TUNING_JOBS_ROOT"] = args.tuning_jobs_root


def build_split(args: argparse.Namespace) -> SplitQueryParams:
    return SplitQueryParams(
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


def build_training_request(args: argparse.Namespace) -> TrainingRequest:
    base = parse_json_object(getattr(args, "hyperparameters_json", None)) or {}
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
        split=build_split(args),
        study=optional_list(args.study),
        dataset=optional_list(args.dataset),
        cell_line=optional_list(args.cell_line),
        pe_system=optional_list(args.pe_system),
        edit_type=optional_list(args.edit_type),
        edit_length=optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=optional_list(args.edit_scope),
        experimental_method=optional_list(args.experimental_method),
        target_context=optional_list(args.target_context),
        scaffold_name=optional_list(args.scaffold_name),
        model_kwargs=parse_json_object(getattr(args, "model_kwargs_json", None)),
        notes=getattr(args, "notes", None),
        device=args.device,
    )


def build_tuning_request(args: argparse.Namespace) -> TuningRequest:
    fixed = parse_json_object(getattr(args, "fixed_hyperparameters_json", None)) or {}
    training = TrainingRequest(
        model_name=args.model,
        dataset_source=getattr(args, "dataset_source", "pe-db"),
        dataset_name=args.dataset_name,
        hyperparameters=fixed or None,
        hyperparameter_mode="replace",
        split=build_split(args),
        study=optional_list(args.study),
        dataset=optional_list(args.dataset),
        cell_line=optional_list(args.cell_line),
        pe_system=optional_list(args.pe_system),
        edit_type=optional_list(args.edit_type),
        edit_length=optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=optional_list(args.edit_scope),
        experimental_method=optional_list(args.experimental_method),
        target_context=optional_list(args.target_context),
        scaffold_name=optional_list(args.scaffold_name),
        model_kwargs=parse_json_object(getattr(args, "model_kwargs_json", None)),
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
        split=build_split(args),
        study=optional_list(args.study),
        dataset=optional_list(args.dataset),
        cell_line=optional_list(args.cell_line),
        pe_system=optional_list(args.pe_system),
        edit_type=optional_list(args.edit_type),
        edit_length=optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=optional_list(args.edit_scope),
        experimental_method=optional_list(args.experimental_method),
        target_context=optional_list(args.target_context),
        scaffold_name=optional_list(args.scaffold_name),
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
    combine_options = parse_json_object(args.combine_options_json) or {}
    return EnsembleRequest(
        ensemble_name=args.ensemble_name,
        combine=args.combine,
        combine_options=combine_options,
        members=[parse_ensemble_member(raw) for raw in args.member],
        split=build_split(args),
        study=optional_list(args.study),
        dataset=optional_list(args.dataset),
        cell_line=optional_list(args.cell_line),
        pe_system=optional_list(args.pe_system),
        edit_type=optional_list(args.edit_type),
        edit_length=optional_list(args.edit_length),
        edit_efficiency_min=getattr(args, "edit_efficiency_min", None),
        edit_efficiency_max=getattr(args, "edit_efficiency_max", None),
        edit_scope=optional_list(args.edit_scope),
        experimental_method=optional_list(args.experimental_method),
        target_context=optional_list(args.target_context),
        scaffold_name=optional_list(args.scaffold_name),
        device=args.device,
    )


def prepare_runtime(argv: Optional[List[str]]) -> Optional[int]:
    """Bootstrap plugins and handle global early-exit flags."""
    apply_default_pe_db_mode()
    early = early_parse(argv)
    if early.list_devices:
        print(format_devices_for_cli())
        return 0
    if early.plugins_root:
        os.environ["PLUGINS_ROOT"] = early.plugins_root
    if early.pe_db_mode:
        os.environ["PE_DB_MODE"] = early.pe_db_mode
    loaded = bootstrap_plugins()
    if loaded:
        print(f"Loaded plugins: {', '.join(loaded)}", file=sys.stderr)
    sync_pe_db_plugins()
    return None

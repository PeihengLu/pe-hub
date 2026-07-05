#!/usr/bin/env python3
"""CLI for Optuna hyperparameter tuning (cluster / SLURM workflows)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pe_common.devices import format_devices_for_cli

from .training.config import presets_root, supported_models, tuning_studies_root
from .training.dataset_key import dataset_preset_key
from .training.hyperparameter_presets import preset_path_for_model, write_dataset_preset
from .training.runner import TrainingError
from .training.schemas import SplitQueryParams, TrainingRequest
from .training.search_spaces import get_search_space, suggest_trial_hyperparameters
from .training.tune_runner import run_tuning_trial
from .train_models import (
    _bootstrap_plugins,
    _early_parse,
    _parse_json_object,
    _sync_pe_db_plugins,
)


def _build_tune_request(args: argparse.Namespace) -> TrainingRequest:
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
    fixed = _parse_json_object(args.fixed_hyperparameters_json) or {}
    return TrainingRequest(
        model_name=args.model,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        hyperparameters=fixed or None,
        hyperparameter_mode="replace",
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
        description=(
            "Tune hyperparameters with Optuna for one model and dataset. "
            "Scheduler settings are not tuned; set them during normal training."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(supported_models()),
        help="Model to tune",
    )
    parser.add_argument("--dataset-source", default="pe-db")
    parser.add_argument("--dataset-name", required=True)
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
    parser.add_argument("--split-strategy", default="holdout_3", choices=["none", "holdout_2", "holdout_3", "cv"])
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--val-pct", type=float, default=0.15)
    parser.add_argument("--test-pct", type=float, default=0.15)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--use-original-fold", action="store_true")
    parser.add_argument("--original-fold-test-value", type=float, default=-1.0)
    parser.add_argument("--split-random-state", type=int, default=42)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--fixed-hyperparameters-json",
        default=None,
        help="JSON object merged into every trial (e.g. load_pretrained, loss_func)",
    )
    parser.add_argument("--model-kwargs-json", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--pe-db-url", default=None)
    parser.add_argument("--pe-db-mode", default=None, choices=["http", "library"])
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--plugins-root", default=None)
    parser.add_argument("--presets-root", default=None, help="Override TRAINING_PRESETS_ROOT")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (default: model__dataset_key)",
    )
    parser.add_argument(
        "--study-storage",
        default=None,
        help="Optuna storage URL (default: sqlite under tuning_studies/)",
    )
    parser.add_argument(
        "--write-preset",
        default=None,
        help="Preset YAML path to update with the best trial (default: presets/<model>.yaml)",
    )
    parser.add_argument(
        "--no-write-preset",
        action="store_true",
        help="Run tuning without writing a dataset preset",
    )
    parser.add_argument(
        "--register-best-weights",
        action="store_true",
        help="After tuning, train once more with best params and register weights",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def _default_study_name(request: TrainingRequest) -> str:
    key = dataset_preset_key(
        study=request.study,
        dataset=request.dataset,
        cell_line=request.cell_line,
        pe_system=request.pe_system,
    )
    slug = (key or request.dataset_name).replace("/", "__")
    return f"{request.model_name.strip().lower()}__{slug}"


def _default_study_storage(study_name: str, root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / f"{study_name}.db"
    return f"sqlite:///{db_path}"


def run_study(args: argparse.Namespace, request: TrainingRequest) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise TrainingError(
            "Optuna is required for tuning. Install with: pip install 'pe-ensemble[tuning]'"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = get_search_space(request.model_name)
    study_name = args.study_name or _default_study_name(request)
    storage = args.study_storage or _default_study_storage(
        study_name,
        tuning_studies_root(),
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=space.direction,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        suggested = suggest_trial_hyperparameters(request.model_name, trial)
        fixed = dict(request.hyperparameters or {})
        trial_params = {**suggested, **fixed}
        result = run_tuning_trial(
            request,
            suggested=trial_params,
            register_weights=False,
        )
        return result.metric

    study.optimize(objective, n_trials=int(args.n_trials))

    best = study.best_trial
    best_params = dict(best.params)
    best_params.update(dict(request.hyperparameters or {}))

    dataset_key = dataset_preset_key(
        study=request.study,
        dataset=request.dataset,
        cell_line=request.cell_line,
        pe_system=request.pe_system,
    )
    if dataset_key is None:
        raise TrainingError(
            "Could not derive a dataset preset key. Provide single --study and --dataset "
            "(and optionally single --cell-line / --pe-system)."
        )

    provenance = {
        "source": "optuna",
        "study_name": study_name,
        "study_storage": storage,
        "best_trial": int(best.number),
        "metric": space.metric,
        "metric_value": float(best.value),
        "direction": space.direction,
        "n_trials": int(args.n_trials),
        "searched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

    preset_path: Optional[Path] = None
    if not args.no_write_preset:
        preset_path = Path(
            args.write_preset
            or preset_path_for_model(request.model_name, root=presets_root())
        )
        write_dataset_preset(
            preset_path,
            model_name=request.model_name,
            dataset_key=dataset_key,
            hyperparameters=best_params,
            provenance=provenance,
        )

    final_payload: Optional[Dict[str, Any]] = None
    if args.register_best_weights:
        from .training.runner import execute_training

        final_request = request.model_copy(
            update={
                "hyperparameters": best_params,
                "hyperparameter_mode": "merge",
                "notes": (request.notes or "optuna best trial"),
            }
        )
        final_payload = execute_training(
            final_request,
            device_id=request.device,
            register_weights=True,
        )

    return {
        "study_name": study_name,
        "study_storage": storage,
        "dataset_preset_key": dataset_key,
        "best_trial": int(best.number),
        "best_value": float(best.value),
        "metric": space.metric,
        "direction": space.direction,
        "best_hyperparameters": best_params,
        "preset_path": str(preset_path) if preset_path else None,
        "final_training": final_payload,
    }


def main(argv: Optional[List[str]] = None) -> int:
    early = _early_parse(argv)
    if early.list_devices:
        print(format_devices_for_cli())
        return 0

    if early.plugins_root:
        os.environ["PLUGINS_ROOT"] = early.plugins_root
    if early.pe_db_mode:
        os.environ["PE_DB_MODE"] = early.pe_db_mode

    loaded = _bootstrap_plugins()
    if loaded:
        print(f"Loaded plugins: {', '.join(loaded)}", file=sys.stderr)
    _sync_pe_db_plugins()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.plugins_root:
        os.environ["PLUGINS_ROOT"] = args.plugins_root
    if args.pe_db_url:
        os.environ["PE_DB_URL"] = args.pe_db_url
    if args.pe_db_mode:
        os.environ["PE_DB_MODE"] = args.pe_db_mode
    if args.weights_root:
        os.environ["WEIGHTS_ROOT"] = args.weights_root
    if args.presets_root:
        os.environ["TRAINING_PRESETS_ROOT"] = args.presets_root

    try:
        request = _build_tune_request(args)
        summary = run_study(args, request)
        print(json.dumps(summary, indent=2))
        return 0
    except TrainingError as exc:
        print(f"Tuning failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

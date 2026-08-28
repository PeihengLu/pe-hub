"""Run Optuna hyperparameter studies (shared by CLI, API, and scheduler)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from .config import local_presets_root, tuning_studies_root
from .dataset_key import dataset_preset_key
from .hyperparameter_presets import preset_path_for_model, write_dataset_preset
from .runner import TrainingError, execute_training
from .search_spaces import (
    get_search_space,
    materialize_hyperparameters,
    suggest_trial_hyperparameters,
)
from .tune_jobs import (
    append_log,
    job_log_context,
    mark_failed,
    mark_running,
    mark_succeeded,
)
from .tune_runner import run_tuning_trial
from .tuning_schemas import TuningRequest

logger = logging.getLogger(__name__)


def _default_study_name(request: TuningRequest) -> str:
    training = request.training
    key = dataset_preset_key(
        study=training.study,
        dataset=training.dataset,
        cell_line=training.cell_line,
        pe_system=training.pe_system,
    )
    slug = (key or training.dataset_name).replace("/", "__")
    return f"{training.model_name.strip().lower()}__{slug}"


def _default_study_storage(study_name: str, root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / f"{study_name}.db"
    return f"sqlite:///{db_path}"


def _log(message: str, *, job_id: Optional[str]) -> None:
    logger.info(message)
    if job_id:
        append_log(job_id, message)


def execute_tuning(
    request: TuningRequest,
    *,
    job_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an Optuna study and optionally register best weights."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise TrainingError(
            "Optuna is required for tuning. Install with: pip install pe-ensemble"
        ) from exc

    training = request.training
    effective_device = device_id or training.device

    def _run() -> Dict[str, Any]:
        if job_id:
            mark_running(job_id)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        space = get_search_space(training.model_name)
        study_name = request.study_name or _default_study_name(request)
        storage = request.study_storage or _default_study_storage(
            study_name,
            tuning_studies_root(),
        )

        _log(f"Starting Optuna study {study_name!r} ({request.n_trials} trials)", job_id=job_id)

        study = optuna.create_study(
            study_name=study_name,
            direction=space.direction,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial: "optuna.Trial") -> float:
            if job_id and is_cancel_requested("tune", job_id):
                raise JobCancelledError(f"Tuning job {job_id} cancelled")
            suggested = suggest_trial_hyperparameters(training.model_name, trial)
            fixed = dict(training.hyperparameters or {})
            trial_params = {**suggested, **fixed}
            result = run_tuning_trial(
                training,
                suggested=trial_params,
                register_weights=False,
            )
            _log(
                f"Trial {trial.number}: {space.metric}={result.metric}",
                job_id=job_id,
            )
            return result.metric

        study.optimize(objective, n_trials=int(request.n_trials))

        best = study.best_trial
        # Optuna best.params omits SearchSpaceSpec.fixed and pre-remap aliases
        # (e.g. OPED ffn_dim). Materialize so presets/final train match trials.
        best_params = materialize_hyperparameters(
            training.model_name,
            dict(best.params),
        )
        best_params.update(dict(training.hyperparameters or {}))

        dataset_key = (
            request.dataset_preset_key
            or dataset_preset_key(
                study=training.study,
                dataset=training.dataset,
                cell_line=training.cell_line,
                pe_system=training.pe_system,
            )
        )
        if dataset_key is None:
            raise TrainingError(
                "Could not derive a dataset preset key. Provide single --study and --dataset "
                "(and optionally single --cell-line / --pe-system), or pass "
                "--dataset-preset-key for merged PE-DB filters."
            )

        provenance = {
            "source": "optuna",
            "study_name": study_name,
            "study_storage": storage,
            "best_trial": int(best.number),
            "metric": space.metric,
            "metric_value": float(best.value),
            "direction": space.direction,
            "n_trials": int(request.n_trials),
            "searched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

        preset_path: Optional[Path] = None
        if not request.no_write_preset:
            preset_path = Path(
                request.write_preset
                or preset_path_for_model(
                    training.model_name, root=local_presets_root()
                )
            )
            write_dataset_preset(
                preset_path,
                model_name=training.model_name,
                dataset_key=dataset_key,
                hyperparameters=best_params,
                provenance=provenance,
            )
            _log(f"Wrote local dataset preset to {preset_path}", job_id=job_id)

        final_payload: Optional[Dict[str, Any]] = None
        if request.register_best_weights:
            final_request = training.model_copy(
                update={
                    "hyperparameters": best_params,
                    "hyperparameter_mode": "merge",
                    "notes": (training.notes or "optuna best trial"),
                    "device": effective_device,
                }
            )
            final_payload = execute_training(
                final_request,
                device_id=effective_device,
                register_weights=True,
            )
            _log("Registered weights from best trial", job_id=job_id)

        summary = {
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
        if job_id:
            mark_succeeded(job_id, summary)
        return summary

    if job_id:
        with job_log_context(job_id):
            try:
                return _run()
            except JobCancelledError:
                from .tune_jobs import mark_cancelled

                mark_cancelled(job_id)
                raise
            except Exception as exc:  # noqa: BLE001
                mark_failed(job_id, str(exc))
                raise
    return _run()

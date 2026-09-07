"""Headless PE Ensemble API (shared by HTTP handlers and the peen CLI)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Tuple

from pe_ensemble.ensemble.combine import COMBINE_METHODS, combine_method_help
from pe_ensemble.ensemble.jobs import (
    create_job as _create_ensemble_job,
    delete_job as _delete_ensemble_job,
    get_job as _get_ensemble_job,
    job_summary as _ensemble_job_summary,
    list_jobs as _list_ensemble_jobs,
    read_logs as _read_ensemble_logs,
    wait_for_job as _wait_for_ensemble_job,
)
from pe_ensemble.ensemble.runner import EnsembleError, execute_ensemble
from pe_ensemble.ensemble.schemas import EnsembleMember, EnsembleRequest
from pe_ensemble.evaluation.benchmark import BenchmarkResolutionError, resolve_evaluation_request
from pe_ensemble.evaluation.jobs import (
    create_job as _create_eval_job,
    delete_job as _delete_eval_job,
    get_job as _get_eval_job,
    job_summary as _eval_job_summary,
    list_jobs as _list_eval_jobs,
    read_logs as _read_eval_logs,
    wait_for_job as _wait_for_eval_job,
)
from pe_ensemble.evaluation.runner import EvaluationError, execute_evaluation
from pe_ensemble.evaluation.schemas import EvaluationRequest
from pe_ensemble.models.registry import model_registry
from pe_ensemble.plugin_loader import load_active_plugins
from pe_ensemble.training.config import (
    enable_cli_pe_db_access,
    is_supported_model,
    jobs_root,
    supported_models,
)
from pe_ensemble.training.data import build_pe_db_filter_params, request_pe_db_filtered
from pe_ensemble.training.hyperparameter_presets import resolve_hyperparameters
from pe_ensemble.training.jobs import (
    create_job as _create_train_job,
    delete_job as _delete_train_job,
    get_job as _get_train_job,
    job_summary as _train_job_summary,
    list_jobs as _list_train_jobs,
    read_logs as _read_train_logs,
    wait_for_job as _wait_for_train_job,
)
from pe_ensemble.training.model_architecture import (
    architecture_from_cli_args,
    merge_training_hyperparameters,
)
from pe_ensemble.training.pe_db_access import PeDbAccessError, reload_pe_db_plugins
from pe_ensemble.training.runner import TrainingError, execute_training
from pe_ensemble.training.schemas import SplitQueryParams, TrainingRequest
from pe_ensemble.training.tune_jobs import (
    create_job as _create_tune_job,
    delete_job as _delete_tune_job,
    get_job as _get_tune_job,
    job_summary as _tune_job_summary,
    list_jobs as _list_tune_jobs,
    read_logs as _read_tune_logs,
    wait_for_job as _wait_for_tune_job,
)
from pe_ensemble.training.tune_study import execute_tuning
from pe_ensemble.training.tuning_schemas import TuningRequest

JobKind = Literal["train", "tune", "evaluate", "ensemble"]

_QUEUED_LABELS = {
    "train": "Training",
    "tune": "Tuning",
    "evaluate": "Evaluation",
    "ensemble": "Ensemble",
}


class PeEnsembleLibraryError(ValueError):
    """Raised when a library operation is given invalid arguments."""


def get_scheduler():
    """Return the process-wide device scheduler (patch point for tests)."""
    from pe_ensemble.compute.device_scheduler import get_scheduler as _get_scheduler

    return _get_scheduler()


def shutdown_compute() -> None:
    """Stop the device scheduler (HTTP process shutdown)."""
    from pe_ensemble.compute.device_scheduler import shutdown_scheduler

    shutdown_scheduler()


def device_snapshot() -> List[Dict[str, Any]]:
    return get_scheduler().device_snapshot()


def require_supported_model(model_name: str) -> str:
    name = model_name.strip().lower()
    if not is_supported_model(name):
        raise PeEnsembleLibraryError("Invalid model name")
    return name


def list_model_catalog() -> List[Dict[str, Any]]:
    return model_registry.list_catalog_entries()


def list_weight_entries(model_name: str) -> List[Dict[str, Any]]:
    return model_registry.list_weight_entries(require_supported_model(model_name))


def validate_weight_selection(model_name: str, weight_id: str) -> None:
    try:
        model_registry.validate_weight_selection(model_name, weight_id)
    except ValueError as exc:
        raise PeEnsembleLibraryError(str(exc)) from exc


def resolve_evaluation(request: EvaluationRequest) -> EvaluationRequest:
    """Validate weights and reconstruct the evaluation split from provenance."""
    model_name = require_supported_model(request.model_name)
    weight_id = request.weights.strip()
    if not weight_id:
        raise PeEnsembleLibraryError("weights is required")
    validate_weight_selection(model_name, weight_id)
    try:
        return resolve_evaluation_request(request)
    except BenchmarkResolutionError as exc:
        raise PeEnsembleLibraryError(str(exc)) from exc


def validate_ensemble_request(request: EnsembleRequest) -> EnsembleRequest:
    if len(request.members) < 2:
        raise PeEnsembleLibraryError("ensemble requires at least two members")
    for member in request.members:
        model_name = require_supported_model(member.model_name)
        weight_id = member.weights.strip()
        if not weight_id:
            raise PeEnsembleLibraryError("Each member requires a weights ID")
        validate_weight_selection(model_name, weight_id)
    return request


def resolve_training_presets(
    model_name: str,
    *,
    study: Optional[str] = None,
    dataset: Optional[str] = None,
    cell_line: Optional[str] = None,
    pe_system: Optional[str] = None,
    hyperparameter_mode: str = "merge",
) -> Dict[str, Any]:
    name = require_supported_model(model_name)
    resolved = resolve_hyperparameters(
        name,
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
        user_overrides=None,
        mode=hyperparameter_mode,
    )
    return {
        "model": name,
        "preset_key": resolved.preset_key,
        "preset_source": resolved.preset_source,
        "hyperparameter_mode": hyperparameter_mode,
        "hyperparameters": resolved.hyperparameters,
    }


def filter_pe_db(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch PE-DB filter/export payload (in-process CLI or HTTP from the web service)."""
    return request_pe_db_filtered(params)


def _job_ops(kind: JobKind) -> Dict[str, Any]:
    return {
        "train": {
            "create": _create_train_job,
            "get": _get_train_job,
            "list": _list_train_jobs,
            "logs": _read_train_logs,
            "wait": _wait_for_train_job,
            "delete": _delete_train_job,
            "summary": _train_job_summary,
            "submit": lambda job_id, request: get_scheduler().submit_training(job_id, request),
        },
        "tune": {
            "create": _create_tune_job,
            "get": _get_tune_job,
            "list": _list_tune_jobs,
            "logs": _read_tune_logs,
            "wait": _wait_for_tune_job,
            "delete": _delete_tune_job,
            "summary": _tune_job_summary,
            "submit": lambda job_id, request: get_scheduler().submit_tuning(job_id, request),
        },
        "evaluate": {
            "create": _create_eval_job,
            "get": _get_eval_job,
            "list": _list_eval_jobs,
            "logs": _read_eval_logs,
            "wait": _wait_for_eval_job,
            "delete": _delete_eval_job,
            "summary": _eval_job_summary,
            "submit": lambda job_id, request: get_scheduler().submit_evaluation(job_id, request),
        },
        "ensemble": {
            "create": _create_ensemble_job,
            "get": _get_ensemble_job,
            "list": _list_ensemble_jobs,
            "logs": _read_ensemble_logs,
            "wait": _wait_for_ensemble_job,
            "delete": _delete_ensemble_job,
            "summary": _ensemble_job_summary,
            "submit": lambda job_id, request: get_scheduler().submit_ensemble(job_id, request),
        },
    }[kind]


def create_job(kind: JobKind, request: Any, *, job_id: Optional[str] = None) -> str:
    return _job_ops(kind)["create"](request, job_id=job_id)


def get_job(kind: JobKind, job_id: str) -> Dict[str, Any]:
    return _job_ops(kind)["get"](job_id)


def list_jobs(kind: JobKind, *, limit: int = 50) -> List[Dict[str, Any]]:
    return _job_ops(kind)["list"](limit=limit)


def read_logs(kind: JobKind, job_id: str, *, offset: int = 0) -> Tuple[str, int]:
    return _job_ops(kind)["logs"](job_id, offset=offset)


def wait_for_job(kind: JobKind, job_id: str, **kwargs: Any) -> Dict[str, Any]:
    return _job_ops(kind)["wait"](job_id, **kwargs)


def delete_job(kind: JobKind, job_id: str) -> None:
    _job_ops(kind)["delete"](job_id)


def job_summary(
    kind: JobKind,
    manifest: Dict[str, Any],
    *,
    include_result: bool = False,
) -> Dict[str, Any]:
    summary = _job_ops(kind)["summary"](manifest).model_dump()
    if include_result and manifest.get("result") is not None:
        summary["result"] = manifest["result"]
    return summary


def list_job_summaries(kind: JobKind, *, limit: int = 50) -> List[Dict[str, Any]]:
    return [job_summary(kind, manifest) for manifest in list_jobs(kind, limit=limit)]


def job_status(kind: JobKind, job_id: str) -> Dict[str, Any]:
    return job_summary(kind, get_job(kind, job_id), include_result=True)


def job_logs(kind: JobKind, job_id: str, *, offset: int = 0) -> Dict[str, Any]:
    manifest = get_job(kind, job_id)
    chunk, next_offset = read_logs(kind, job_id, offset=offset)
    return {
        "job_id": job_id,
        "status": manifest["status"],
        "offset": offset,
        "next_offset": next_offset,
        "log": chunk,
    }


def queued_message(kind: JobKind, manifest: Dict[str, Any]) -> str:
    label = _QUEUED_LABELS[kind]
    position = manifest.get("queue_position")
    if position:
        return f"{label} job queued (position {position})"
    return f"{label} job started"


def submit_job(kind: JobKind, job_id: str, request: Any) -> None:
    _job_ops(kind)["submit"](job_id, request)


def ensure_job(kind: JobKind, request: Any, *, job_id: Optional[str] = None) -> str:
    """Create a job, or reuse ``job_id`` when that record already exists."""
    if job_id:
        try:
            get_job(kind, job_id)
            return job_id
        except FileNotFoundError:
            return create_job(kind, request, job_id=job_id)
    return create_job(kind, request)


def queue_job(
    kind: JobKind,
    request: Any,
    *,
    job_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Create the job record and submit it to the device scheduler."""
    created = create_job(kind, request, job_id=job_id)
    submit_job(kind, created, request)
    return created, get_job(kind, created)


def load_training_request(job_id: str) -> TrainingRequest:
    path = jobs_root() / job_id / "request.json"
    with path.open(encoding="utf-8") as handle:
        return TrainingRequest.model_validate(json.load(handle))


def rerun_training_job(job_id: str) -> Dict[str, Any]:
    """Submit an existing queued/failed training job and wait for completion."""
    manifest = get_job("train", job_id)
    if manifest["status"] not in ("queued", "failed"):
        raise PeEnsembleLibraryError(f"Job {job_id} is already {manifest['status']}")
    request = load_training_request(job_id)
    submit_job("train", job_id, request)
    return wait_for_job("train", job_id)


def begin_kill(kind: JobKind, job_id: str) -> Optional[Dict[str, Any]]:
    from pe_ensemble.compute.job_lifecycle import begin_job_kill

    return begin_job_kill(kind, job_id, get_job=lambda jid: get_job(kind, jid))


def finalize_kill(kind: JobKind, job_id: str) -> None:
    from pe_ensemble.compute.job_lifecycle import finalize_job_kill

    finalize_job_kill(
        kind,
        job_id,
        get_job=lambda jid: get_job(kind, jid),
        delete_job=lambda jid: delete_job(kind, jid),
    )


__all__ = [
    "COMBINE_METHODS",
    "BenchmarkResolutionError",
    "EnsembleError",
    "EnsembleMember",
    "EnsembleRequest",
    "EvaluationError",
    "EvaluationRequest",
    "JobKind",
    "PeDbAccessError",
    "PeEnsembleLibraryError",
    "SplitQueryParams",
    "TrainingError",
    "TrainingRequest",
    "TuningRequest",
    "architecture_from_cli_args",
    "begin_kill",
    "build_pe_db_filter_params",
    "combine_method_help",
    "create_job",
    "delete_job",
    "device_snapshot",
    "enable_cli_pe_db_access",
    "ensure_job",
    "execute_ensemble",
    "execute_evaluation",
    "execute_training",
    "execute_tuning",
    "filter_pe_db",
    "finalize_kill",
    "get_job",
    "get_scheduler",
    "is_supported_model",
    "job_logs",
    "job_status",
    "job_summary",
    "jobs_root",
    "list_job_summaries",
    "list_jobs",
    "list_model_catalog",
    "list_weight_entries",
    "load_active_plugins",
    "load_training_request",
    "merge_training_hyperparameters",
    "model_registry",
    "queue_job",
    "queued_message",
    "read_logs",
    "reload_pe_db_plugins",
    "require_supported_model",
    "rerun_training_job",
    "resolve_evaluation",
    "resolve_training_presets",
    "shutdown_compute",
    "submit_job",
    "supported_models",
    "validate_ensemble_request",
    "validate_weight_selection",
    "wait_for_job",
]

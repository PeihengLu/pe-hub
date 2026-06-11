"""Execute a training job (shared by API and CLI)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from pe_common.devices import AUTO_DEVICE, cuda_index_from_device, resolve_device, resolve_device_id
from pe_common.splits import exclude_test_partition

from ..models import weights_registry
from ..models.model_factory import ModelFactory
from .config import MODEL_FORMAT, SUPPORTED_MODELS
from .data import fetch_training_dataframe, normalize_filter_param
from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from .jobs import append_log, job_log_context, mark_cancelled, mark_failed, mark_running, mark_succeeded
from .progress_log import JOB_CANCEL_CHECK_KEY, JOB_PROGRESS_LOG_KEY, tee_stream_to_log
from .schemas import TrainingRequest

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when training input or execution fails."""


def _training_metadata_from_request(
    request: TrainingRequest,
    *,
    n_rows: int,
    train_result: Dict[str, Any],
    device_id: str,
) -> Dict[str, Any]:
    filters = {
        key: normalize_filter_param(getattr(request, key))
        for key in (
            "study",
            "dataset",
            "cell_line",
            "pe_system",
            "edit_type",
            "edit_length",
            "edit_scope",
            "experimental_method",
            "target_context",
            "scaffold_name",
        )
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    metrics: Dict[str, Any] = {}
    if "validation_metrics" in train_result:
        metrics["validation"] = train_result["validation_metrics"]
    elif "val_pearson" in train_result:
        metrics["validation"] = {
            "pearson": train_result.get("val_pearson"),
            "spearman": train_result.get("val_spearman"),
        }
    return {
        "training": {
            "dataset_source": request.dataset_source,
            "dataset_name": request.dataset_name,
            "filters": filters,
            "split": request.split.model_dump(),
            "hyperparameters": request.hyperparameters or {},
            "model_kwargs": request.model_kwargs or {},
            "n_train_rows": n_rows,
            "device": device_id,
        },
        "metrics": metrics,
        "notes": request.notes,
    }


def _merge_hyperparameters(request: TrainingRequest, device: torch.device) -> Dict[str, Any]:
    hyperparameters = dict(request.hyperparameters or {})
    if request.model_name.strip().lower() == "pridict2" and "gpu_index" not in hyperparameters:
        hyperparameters["gpu_index"] = cuda_index_from_device(device)
    return hyperparameters


def execute_training(
    request: TrainingRequest,
    *,
    job_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run training end-to-end and register weights."""
    model_name = request.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise TrainingError(f"Invalid model name: {model_name}")

    resolved_device_id = resolve_device_id(device_id or request.device or AUTO_DEVICE)
    device = resolve_device(resolved_device_id)

    def _log(message: str) -> None:
        logger.info(message)
        if job_id:
            append_log(job_id, message)

    def _raise_if_cancelled() -> None:
        if job_id and is_cancel_requested("train", job_id):
            raise JobCancelledError(f"Training job {job_id} cancelled")

    def _progress_log(message: str) -> None:
        _raise_if_cancelled()
        _log(message)

    context = job_log_context(job_id) if job_id else _null_context()
    with context:
        if job_id:
            mark_running(job_id)
        _log(
            f"Starting training for model={model_name} dataset={request.dataset_name} "
            f"device={resolved_device_id}"
        )

        try:
            _raise_if_cancelled()
            train_df = fetch_training_dataframe(
                request,
                MODEL_FORMAT[model_name],
                progress_log=_progress_log,
            )
            if train_df.empty:
                raise TrainingError("No training data resolved.")

            train_df = exclude_test_partition(train_df)
            if train_df.empty:
                raise TrainingError("No non-test rows available for training.")

            _log(f"Resolved {len(train_df)} training rows")

            _raise_if_cancelled()
            model = ModelFactory.create_model(
                model_name,
                device=device,
                **(request.model_kwargs or {}),
            )
            if model_name == "oped":
                _progress_log(f"Tokenizing {len(train_df)} OPED sequences for training...")
                train_df = model.prepare_data(train_df)
                _progress_log(f"Encoded {len(train_df)} rows; starting training loop")

            hyperparameters = _merge_hyperparameters(request, device)
            hyperparameters[JOB_PROGRESS_LOG_KEY] = _progress_log
            hyperparameters[JOB_CANCEL_CHECK_KEY] = _raise_if_cancelled
            _raise_if_cancelled()
            if model_name == "pridict2":
                with tee_stream_to_log(
                    _progress_log if job_id else None,
                    cancel_check=_raise_if_cancelled if job_id else None,
                ):
                    result = model.train(train_df, hyperparameters=hyperparameters)
            else:
                result = model.train(train_df, hyperparameters=hyperparameters)
            metadata = _training_metadata_from_request(
                request,
                n_rows=int(len(train_df)),
                train_result=result,
                device_id=resolved_device_id,
            )
            weights_id = weights_registry.register_trained_model(
                model_name,
                model,
                metadata=metadata,
                notes=request.notes,
            )
            entry = weights_registry.get_manifest(model_name, weights_id)
        except JobCancelledError:
            if job_id:
                mark_cancelled(job_id)
            raise
        except Exception as exc:
            if job_id:
                mark_failed(job_id, str(exc))
            raise

        payload = {
            "model": model_name,
            "status": "success",
            "split_strategy": request.split.split_strategy,
            "n_rows": int(len(train_df)),
            "device": resolved_device_id,
            "weights_id": weights_id,
            "weights_label": entry.get("label"),
            "result": result,
        }
        _log(f"Training succeeded; weights_id={weights_id}")
        if job_id:
            mark_succeeded(job_id, payload)
        return payload


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


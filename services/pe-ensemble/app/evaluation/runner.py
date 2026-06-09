"""Execute an evaluation job (shared by API and workers)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pe_common.devices import AUTO_DEVICE, resolve_device, resolve_device_id

from ..models.model_factory import ModelFactory
from ..training.config import MODEL_FORMAT, SUPPORTED_MODELS
from ..training.data import fetch_model_format_dataframe
from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from .jobs import append_log, job_log_context, mark_cancelled, mark_failed, mark_running, mark_succeeded
from .schemas import EvaluationRequest

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation input or execution fails."""


def execute_evaluation(
    request: EvaluationRequest,
    *,
    job_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    model_name = request.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise EvaluationError(f"Invalid model name: {model_name}")

    resolved_device_id = resolve_device_id(device_id or request.device or AUTO_DEVICE)
    device = resolve_device(resolved_device_id)
    model_format = MODEL_FORMAT[model_name]

    def _log(message: str) -> None:
        logger.info(message)
        if job_id:
            append_log(job_id, message)

    def _raise_if_cancelled() -> None:
        if job_id and is_cancel_requested("evaluate", job_id):
            raise JobCancelledError(f"Evaluation job {job_id} cancelled")

    context = job_log_context(job_id) if job_id else _null_context()
    with context:
        if job_id:
            mark_running(job_id)
        _log(
            f"Starting evaluation for model={model_name} benchmark={request.benchmark_name} "
            f"device={resolved_device_id}"
        )

        try:
            _raise_if_cancelled()
            test_df = fetch_model_format_dataframe(
                model_format=model_format,
                split=request.split,
                records=request.records,
                study=request.study,
                dataset=request.dataset,
                cell_line=request.cell_line,
                pe_system=request.pe_system,
                edit_type=request.edit_type,
                edit_length=request.edit_length,
                edit_efficiency_min=request.edit_efficiency_min,
                edit_efficiency_max=request.edit_efficiency_max,
                edit_scope=request.edit_scope,
                experimental_method=request.experimental_method,
                target_context=request.target_context,
                scaffold_name=request.scaffold_name,
                evaluation=True,
            )
            if test_df.empty:
                raise EvaluationError("No test data resolved for evaluation.")

            _log(f"Resolved {len(test_df)} test rows")

            _raise_if_cancelled()
            model = ModelFactory.create_model(
                model_name,
                device=device,
                **(request.model_kwargs or {}),
            )

            if model_name == "oped":
                prepared = model.prepare_data(test_df)
                _raise_if_cancelled()
                metrics = model.evaluate(prepared, weights=request.weights)
            else:
                _raise_if_cancelled()
                metrics = model.evaluate(test_df, weights=request.weights)
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
            "benchmark_name": request.benchmark_name,
            "weights": request.weights,
            "device": resolved_device_id,
            "n_samples": int(len(test_df)),
            "metrics": metrics,
        }
        _log(f"Evaluation succeeded; n_samples={payload['n_samples']}")
        if job_id:
            mark_succeeded(job_id, payload)
        return payload


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

"""Execute an evaluation job (shared by API and workers)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pe_common.filter_params import filter_kwargs_from_mapping
from pe_common.devices import AUTO_DEVICE, resolve_device, resolve_device_id

from ..models.model_factory import ModelFactory
from ..training.config import is_supported_model, model_format_for
from ..training.data import ModelFormatFetchResult, fetch_model_format_result
from ..training.progress_log import model_run_stream_context
from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from .benchmark import BenchmarkResolutionError, resolve_evaluation_request
from .jobs import append_log, job_log_context, mark_cancelled, mark_failed, mark_running, mark_skipped, mark_succeeded
from ..models import weights_registry
from .leakage import (
    REASON_TRAIN_TEST_OVERLAP,
    assess_leakage,
    exclude_overlapping_loci,
    leak_error_payload,
    restrict_to_author_holdout_rows,
)
from .schemas import EvaluationRequest

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation input or execution fails."""


def _format_skipped_datasheet(entry: Dict[str, Any]) -> str:
    return (
        f"{entry.get('study')}/{entry.get('dataset')} "
        f"({entry.get('cell_line')}-{entry.get('pe_system')}): "
        f"{entry.get('reason', 'skipped')}"
    )


def _evaluation_skip_reason(fetch: ModelFormatFetchResult) -> Optional[str]:
    """Return a user-facing skip reason when no test rows can be evaluated."""
    if fetch.skipped:
        return "; ".join(_format_skipped_datasheet(entry) for entry in fetch.skipped)
    if fetch.partition_error:
        return fetch.partition_error
    if fetch.total_records > 0:
        return "Converted records exist but the test partition is empty."
    return None


def execute_evaluation(
    request: EvaluationRequest,
    *,
    job_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    model_name = request.model_name.strip().lower()
    if not is_supported_model(model_name):
        raise EvaluationError(f"Invalid model name: {model_name}")

    resolved_device_id = resolve_device_id(device_id or request.device or AUTO_DEVICE)
    device = resolve_device(resolved_device_id)
    model_format = model_format_for(model_name)

    def _log(message: str) -> None:
        logger.info(message)
        if job_id:
            append_log(job_id, message)

    def _raise_if_cancelled() -> None:
        if job_id and is_cancel_requested("evaluate", job_id):
            raise JobCancelledError(f"Evaluation job {job_id} cancelled")

    def _progress_log(message: str) -> None:
        _raise_if_cancelled()
        _log(message)

    context = job_log_context(job_id) if job_id else _null_context()
    with context:
        if job_id:
            mark_running(job_id)
        try:
            request = resolve_evaluation_request(request)
        except BenchmarkResolutionError as exc:
            if job_id:
                mark_failed(job_id, str(exc))
            raise EvaluationError(str(exc)) from exc

        _log(
            f"Starting evaluation for model={model_name} benchmark={request.benchmark_name} "
            f"device={resolved_device_id}"
        )
        if request.auto_training_benchmark:
            _log("Using training-recorded filters and split for test-set selection")

        try:
            _raise_if_cancelled()
            fetch = fetch_model_format_result(
                model_format=model_format,
                split=request.split,
                records=request.records,
                evaluation=True,
                progress_log=_progress_log,
                **filter_kwargs_from_mapping(request.model_dump()),
            )
            test_df = fetch.df
            if test_df.empty:
                skip_reason = _evaluation_skip_reason(fetch)
                if skip_reason:
                    message = f"Skipped evaluation: {skip_reason}"
                    _log(message)
                    payload = {
                        "model": model_name,
                        "benchmark_name": request.benchmark_name,
                        "weights": request.weights,
                        "device": resolved_device_id,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "skipped_datasheets": fetch.skipped,
                        "n_samples": 0,
                        "metrics": None,
                    }
                    if job_id:
                        mark_skipped(job_id, payload, reason=skip_reason)
                    return payload
                raise EvaluationError("No test data resolved for evaluation.")

            _log(f"Resolved {len(test_df)} test rows")
            test_df, n_unlabeled_dropped = restrict_to_author_holdout_rows(test_df)
            if n_unlabeled_dropped:
                _log(
                    f"Dropped {n_unlabeled_dropped} unlabeled rows from an "
                    f"author-fold test; evaluating {len(test_df)} remaining rows"
                )

            leak = None
            leak_exclusion_warning: Optional[Dict[str, Any]] = None
            leak = assess_leakage(
                test_df=test_df,
                split=request.split,
                model=model_name,
                weights_id=request.weights,
                eval_datasets=request.dataset,
            )
            if leak is not None and leak.is_leak:
                if (
                    leak.reason == REASON_TRAIN_TEST_OVERLAP
                    and not request.allow_data_leak
                ):
                    training_loci = weights_registry.load_training_loci(
                        model_name, request.weights
                    ) or set()
                    exclusion = exclude_overlapping_loci(test_df, training_loci)
                    if exclusion is not None and not exclusion.is_empty:
                        test_df = exclusion.filtered_df
                        leak_exclusion_warning = exclusion.warning_payload()
                        _log(leak_exclusion_warning["message"])
                    else:
                        payload = leak_error_payload(
                            leak,
                            model=model_name,
                            benchmark_name=request.benchmark_name,
                            weights=request.weights,
                            device_id=resolved_device_id,
                            n_samples=int(len(test_df)),
                        )
                        if exclusion is not None and exclusion.is_empty:
                            payload["leak"] = {
                                **payload.get("leak", {}),
                                **exclusion.warning_payload(),
                                "message": (
                                    "All evaluation target loci overlap this model's "
                                    "training data; nothing remains after exclusion."
                                ),
                            }
                        message = (
                            f"Aborting evaluation: potential data leak ({leak.reason}). "
                            f"{payload['leak'].get('message', leak.detail.get('message', ''))}"
                        )
                        _log(message)
                        if job_id:
                            mark_failed(
                                job_id,
                                f"{payload['error_type']}: {leak.reason}",
                                result=payload,
                            )
                        return payload
                elif not request.allow_data_leak:
                    payload = leak_error_payload(
                        leak,
                        model=model_name,
                        benchmark_name=request.benchmark_name,
                        weights=request.weights,
                        device_id=resolved_device_id,
                        n_samples=int(len(test_df)),
                    )
                    message = (
                        f"Aborting evaluation: potential data leak ({leak.reason}). "
                        f"{leak.detail.get('message', '')}"
                    )
                    _log(message)
                    if job_id:
                        mark_failed(
                            job_id,
                            f"{payload['error_type']}: {leak.reason}",
                            result=payload,
                        )
                    return payload

            _raise_if_cancelled()
            model = ModelFactory.create_model(model_name, device=device)
            prepared = getattr(model, "prepare_evaluation_frame", lambda df: df)(test_df)
            _raise_if_cancelled()
            with model_run_stream_context(
                model,
                _progress_log if job_id else None,
                cancel_check=_raise_if_cancelled if job_id else None,
            ):
                metrics = model.evaluate(prepared, weights=request.weights)
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
            "auto_training_benchmark": request.auto_training_benchmark,
        }
        if leak_exclusion_warning is not None:
            payload["leak_warning"] = leak_exclusion_warning
            _log(
                "Warning: proceeded after excluding training-overlapping test loci; "
                "see leak_warning for counts."
            )
        elif leak is not None and leak.is_leak and request.allow_data_leak:
            payload["leak_warning"] = {"reason": leak.reason, **leak.detail}
            _log(
                f"Warning: proceeded despite potential data leak ({leak.reason}); "
                "metrics may be optimistic."
            )
        _log(f"Evaluation succeeded; n_samples={payload['n_samples']}")
        if job_id:
            mark_succeeded(job_id, payload)
        return payload


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

"""Execute an ensemble evaluation job (shared by API and workers)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pe_common.devices import AUTO_DEVICE, resolve_device, resolve_device_id
from pe_common.training import regression_metrics

from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from ..models.model_factory import ModelFactory
from ..training.config import is_supported_model, model_format_for
from ..training.data import ModelFormatFetchResult, fetch_model_format_result
from .combine import combine_predictions
from .jobs import append_log, job_log_context, mark_cancelled, mark_failed, mark_running, mark_skipped, mark_succeeded
from .schemas import EnsembleMember, EnsembleRequest

logger = logging.getLogger(__name__)


class EnsembleError(Exception):
    """Raised when ensemble input or execution fails."""


def _format_skipped_datasheet(entry: Dict[str, Any]) -> str:
    return (
        f"{entry.get('study')}/{entry.get('dataset')} "
        f"({entry.get('cell_line')}-{entry.get('pe_system')}): "
        f"{entry.get('reason', 'skipped')}"
    )


def _ensemble_skip_reason(fetch: ModelFormatFetchResult) -> Optional[str]:
    if fetch.skipped:
        return "; ".join(_format_skipped_datasheet(entry) for entry in fetch.skipped)
    if fetch.partition_error:
        return fetch.partition_error
    if fetch.total_records > 0:
        return "Converted records exist but the test partition is empty."
    return None


def _extract_label_column(df: pd.DataFrame) -> np.ndarray:
    for column in ("editing_efficiency", "PE_efficiency", "Efficiency", "averageedited"):
        if column in df.columns:
            return df[column].astype(float).to_numpy()
    raise EnsembleError(
        "Could not resolve ground-truth labels from standardized test data. "
        "Expected editing_efficiency, PE_efficiency, Efficiency, or averageedited."
    )


def _resolve_combine_options(request: EnsembleRequest) -> Dict[str, Any]:
    options = dict(request.combine_options)
    if request.combine == "weighted_mean" and "weights" not in options:
        member_weights = [member.member_weight for member in request.members]
        if any(weight is None for weight in member_weights):
            raise EnsembleError("weighted_mean requires member_weight on each member")
        options["weights"] = [float(weight) for weight in member_weights]  # type: ignore[arg-type]
    return options


def _fetch_partition(
    request: EnsembleRequest,
    *,
    model_format: str,
    progress_log: Callable[[str], None],
) -> ModelFormatFetchResult:
    return fetch_model_format_result(
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
        progress_log=progress_log,
    )


def _align_member_matrix(
    std_df: pd.DataFrame,
    member_frames: List[pd.DataFrame],
    member_predictions: List[List[float]],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    lengths = [len(std_df)] + [len(frame) for frame in member_frames] + [
        len(preds) for preds in member_predictions
    ]
    n_aligned = min(lengths)
    alignment = {
        "std_rows": int(len(std_df)),
        "member_rows": [int(len(frame)) for frame in member_frames],
        "rows_after_alignment": int(n_aligned),
    }
    if len(set(lengths)) != 1:
        alignment["truncated"] = True
        alignment["warning"] = (
            "Member row counts differed; predictions were truncated to the shortest partition."
        )
    y_true = _extract_label_column(std_df.iloc[:n_aligned])
    matrix = np.column_stack(
        [np.asarray(predictions[:n_aligned], dtype=float) for predictions in member_predictions]
    )
    return y_true, matrix, alignment


def _predict_member(
    member: EnsembleMember,
    test_df: pd.DataFrame,
    *,
    device,
) -> List[float]:
    model_name = member.model_name.strip().lower()
    model = ModelFactory.create_model(model_name, device=device)
    model.load_weights_by_name(member.weights)

    if model_name == "oped":
        prepared = model.prepare_data(test_df)
        return model.predict(prepared)
    if model_name == "pridict2":
        dloader = model.prepare_data(test_df, y_ref=["averageedited"])
        return model.predict(dloader)
    feature_df = test_df.copy()
    for column in ("Efficiency", "PE_efficiency", "averageedited"):
        if column in feature_df.columns:
            feature_df = feature_df.drop(columns=[column])
    prepared = model.prepare_data(feature_df)
    return model.predict(prepared)


def execute_ensemble(
    request: EnsembleRequest,
    *,
    job_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_device_id = resolve_device_id(device_id or request.device or AUTO_DEVICE)
    device = resolve_device(resolved_device_id)

    for member in request.members:
        model_name = member.model_name.strip().lower()
        if not is_supported_model(model_name):
            raise EnsembleError(f"Invalid model name: {member.model_name}")
        if not member.weights.strip():
            raise EnsembleError(f"weights is required for member {member.model_name}")

    def _log(message: str) -> None:
        logger.info(message)
        if job_id:
            append_log(job_id, message)

    def _raise_if_cancelled() -> None:
        if job_id and is_cancel_requested("ensemble", job_id):
            raise JobCancelledError(f"Ensemble job {job_id} cancelled")

    def _progress_log(message: str) -> None:
        _raise_if_cancelled()
        _log(message)

    context = job_log_context(job_id) if job_id else _null_context()
    with context:
        if job_id:
            mark_running(job_id)
        member_labels = ", ".join(
            f"{member.model_name}:{member.weights}" for member in request.members
        )
        _log(
            f"Starting ensemble={request.ensemble_name} combine={request.combine} "
            f"device={resolved_device_id} members=[{member_labels}]"
        )

        try:
            _raise_if_cancelled()
            std_fetch = _fetch_partition(request, model_format="std", progress_log=_progress_log)
            std_df = std_fetch.df
            if std_df.empty:
                skip_reason = _ensemble_skip_reason(std_fetch)
                if skip_reason:
                    message = f"Skipped ensemble: {skip_reason}"
                    _log(message)
                    payload = {
                        "ensemble_name": request.ensemble_name,
                        "combine": request.combine,
                        "device": resolved_device_id,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "skipped_datasheets": std_fetch.skipped,
                        "n_samples": 0,
                        "metrics": None,
                        "member_metrics": [],
                    }
                    if job_id:
                        mark_skipped(job_id, payload, reason=skip_reason)
                    return payload
                raise EnsembleError("No test data resolved for ensemble evaluation.")

            _log(f"Resolved {len(std_df)} standardized test rows for labels")

            member_frames: List[pd.DataFrame] = []
            member_predictions: List[List[float]] = []
            member_metrics: List[Dict[str, Any]] = []

            for index, member in enumerate(request.members, start=1):
                _raise_if_cancelled()
                model_name = member.model_name.strip().lower()
                model_format = model_format_for(model_name)
                _progress_log(
                    f"Fetching member {index}/{len(request.members)} "
                    f"({model_name}, format={model_format})"
                )
                member_fetch = _fetch_partition(
                    request,
                    model_format=model_format,
                    progress_log=_progress_log,
                )
                member_df = member_fetch.df
                if member_df.empty:
                    reason = _ensemble_skip_reason(member_fetch) or (
                        f"No test rows for member {model_name}"
                    )
                    raise EnsembleError(reason)

                _progress_log(
                    f"Predicting with member {index}/{len(request.members)} "
                    f"({model_name}, weights={member.weights})"
                )
                predictions = _predict_member(member, member_df, device=device)
                member_frames.append(member_df)
                member_predictions.append(predictions)

            y_true, prediction_matrix, alignment = _align_member_matrix(
                std_df,
                member_frames,
                member_predictions,
            )
            if prediction_matrix.shape[0] < 2:
                raise EnsembleError("Need at least two aligned test rows for ensemble metrics.")

            combine_options = _resolve_combine_options(request)
            combined = combine_predictions(
                prediction_matrix,
                request.combine,
                options=combine_options,
            )
            combined_metrics = regression_metrics(y_true, combined.tolist())
            for member_index, member in enumerate(request.members):
                member_metrics.append(
                    {
                        "model_name": member.model_name.strip().lower(),
                        "weights": member.weights,
                        "metrics": regression_metrics(
                            y_true,
                            prediction_matrix[:, member_index].tolist(),
                        ),
                    }
                )

            payload = {
                "ensemble_name": request.ensemble_name,
                "combine": request.combine,
                "combine_options": combine_options,
                "device": resolved_device_id,
                "n_samples": int(prediction_matrix.shape[0]),
                "metrics": combined_metrics,
                "member_metrics": member_metrics,
                "alignment": alignment,
                "members": [
                    {
                        "model_name": member.model_name.strip().lower(),
                        "weights": member.weights,
                    }
                    for member in request.members
                ],
            }
            _log(
                f"Ensemble succeeded; n_samples={payload['n_samples']} "
                f"pearson={combined_metrics.get('pearson')}"
            )
            if job_id:
                mark_succeeded(job_id, payload)
            return payload
        except JobCancelledError:
            if job_id:
                mark_cancelled(job_id)
            raise
        except Exception as exc:
            if job_id:
                mark_failed(job_id, str(exc))
            raise


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False

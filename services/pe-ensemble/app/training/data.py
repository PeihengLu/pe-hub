"""Resolve training data from PE-DB or inline records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

from .config import pe_db_filter_timeout, pe_db_url
from .conversion_progress import pe_db_filter_progress
from .schemas import FilterScalar, FilterValue, SplitQueryParams, TrainingRequest


def normalize_filter_param(value: Optional[FilterValue]) -> Optional[List[FilterScalar]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    return [value]


def build_pe_db_filter_params(
    *,
    model_format: str,
    split: Optional[SplitQueryParams] = None,
    study: Optional[FilterValue] = None,
    dataset: Optional[FilterValue] = None,
    cell_line: Optional[FilterValue] = None,
    pe_system: Optional[FilterValue] = None,
    edit_type: Optional[FilterValue] = None,
    edit_length: Optional[FilterValue] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[FilterValue] = None,
    experimental_method: Optional[FilterValue] = None,
    target_context: Optional[FilterValue] = None,
    scaffold_name: Optional[FilterValue] = None,
) -> Dict[str, Any]:
    split = split or SplitQueryParams()
    params: Dict[str, Any] = {
        "format": model_format,
        "split_strategy": split.split_strategy,
        "use_original_fold": split.use_original_fold,
        "original_fold_test_value": split.original_fold_test_value,
        "split_random_state": split.split_random_state,
        "merge": split.merge,
    }
    if split.train_pct is not None:
        params["train_pct"] = split.train_pct
    if split.val_pct is not None:
        params["val_pct"] = split.val_pct
    if split.test_pct is not None:
        params["test_pct"] = split.test_pct
    if split.cv_folds is not None:
        params["cv_folds"] = split.cv_folds
    for name, value in (
        ("study", study),
        ("dataset", dataset),
        ("cell_line", cell_line),
        ("pe_system", pe_system),
        ("edit_type", edit_type),
        ("edit_length", edit_length),
        ("edit_scope", edit_scope),
        ("experimental_method", experimental_method),
        ("target_context", target_context),
        ("scaffold_name", scaffold_name),
    ):
        normalized = normalize_filter_param(value)
        if normalized is not None:
            params[name] = normalized
    if edit_efficiency_min is not None:
        params["edit_efficiency_min"] = edit_efficiency_min
    if edit_efficiency_max is not None:
        params["edit_efficiency_max"] = edit_efficiency_max
    return params


def request_pe_db_filtered(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(
        f"{pe_db_url()}/api/filter",
        params=params,
        timeout=pe_db_filter_timeout(),
    )
    response.raise_for_status()
    return response.json()


def filtered_payload_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(group["records"])
        for group in payload.get("groups", [])
        if group.get("records")
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@dataclass
class ModelFormatFetchResult:
    """PE-DB filter response materialized as a dataframe plus export metadata."""

    df: pd.DataFrame
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    total_records: int = 0
    partition_error: Optional[str] = None


def fetch_model_format_result(
    *,
    model_format: str,
    split: SplitQueryParams,
    records: Optional[List[Dict[str, Any]]] = None,
    study: Optional[FilterValue] = None,
    dataset: Optional[FilterValue] = None,
    cell_line: Optional[FilterValue] = None,
    pe_system: Optional[FilterValue] = None,
    edit_type: Optional[FilterValue] = None,
    edit_length: Optional[FilterValue] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[FilterValue] = None,
    experimental_method: Optional[FilterValue] = None,
    target_context: Optional[FilterValue] = None,
    scaffold_name: Optional[FilterValue] = None,
    evaluation: bool = False,
    progress_log: Optional[Callable[[str], None]] = None,
) -> ModelFormatFetchResult:
    from pe_common.splits import select_evaluation_partition

    if records is not None:
        df = pd.DataFrame(records)
        if evaluation and split.split_strategy != "none":
            try:
                df = select_evaluation_partition(df, require_test=True)
            except ValueError as exc:
                return ModelFormatFetchResult(
                    pd.DataFrame(),
                    partition_error=str(exc),
                )
        return ModelFormatFetchResult(df)

    params = build_pe_db_filter_params(
        model_format=model_format,
        split=split,
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
        edit_type=edit_type,
        edit_length=edit_length,
        edit_efficiency_min=edit_efficiency_min,
        edit_efficiency_max=edit_efficiency_max,
        edit_scope=edit_scope,
        experimental_method=experimental_method,
        target_context=target_context,
        scaffold_name=scaffold_name,
    )
    if progress_log is not None:
        progress_log(f"Fetching {model_format} data from PE-DB (conversion may take a while)...")
    with pe_db_filter_progress(params, progress_log=progress_log, request_fn=request_pe_db_filtered) as payload:
        skipped = list(payload.get("skipped") or [])
        total_records = int(payload.get("total_records") or 0)
        df = filtered_payload_to_dataframe(payload)
    if evaluation and split.split_strategy != "none":
        try:
            df = select_evaluation_partition(df, require_test=True)
        except ValueError as exc:
            return ModelFormatFetchResult(
                pd.DataFrame(),
                skipped=skipped,
                total_records=total_records,
                partition_error=str(exc),
            )
    return ModelFormatFetchResult(
        df,
        skipped=skipped,
        total_records=total_records,
    )


def fetch_model_format_dataframe(
    *,
    model_format: str,
    split: SplitQueryParams,
    records: Optional[List[Dict[str, Any]]] = None,
    study: Optional[FilterValue] = None,
    dataset: Optional[FilterValue] = None,
    cell_line: Optional[FilterValue] = None,
    pe_system: Optional[FilterValue] = None,
    edit_type: Optional[FilterValue] = None,
    edit_length: Optional[FilterValue] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[FilterValue] = None,
    experimental_method: Optional[FilterValue] = None,
    target_context: Optional[FilterValue] = None,
    scaffold_name: Optional[FilterValue] = None,
    evaluation: bool = False,
) -> pd.DataFrame:
    return fetch_model_format_result(
        model_format=model_format,
        split=split,
        records=records,
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
        edit_type=edit_type,
        edit_length=edit_length,
        edit_efficiency_min=edit_efficiency_min,
        edit_efficiency_max=edit_efficiency_max,
        edit_scope=edit_scope,
        experimental_method=experimental_method,
        target_context=target_context,
        scaffold_name=scaffold_name,
        evaluation=evaluation,
    ).df


def fetch_training_dataframe(
    request: TrainingRequest,
    model_format: str,
    *,
    progress_log: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
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
        evaluation=False,
        progress_log=progress_log,
    ).df

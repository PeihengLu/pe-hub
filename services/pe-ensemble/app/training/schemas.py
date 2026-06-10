"""Shared training request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]
FilterScalar = Union[str, int]
FilterValue = Union[FilterScalar, List[FilterScalar]]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def default_training_split() -> "SplitQueryParams":
    return SplitQueryParams(
        split_strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
    )


class SplitQueryParams(BaseModel):
    split_strategy: SplitStrategy = "none"
    train_pct: Optional[float] = None
    val_pct: Optional[float] = None
    test_pct: Optional[float] = None
    cv_folds: Optional[int] = None
    use_original_fold: bool = False
    original_fold_test_value: float = -1.0
    split_random_state: int = 42
    merge: bool = False


class TrainingRequest(BaseModel):
    model_name: str
    dataset_source: str
    dataset_name: str
    hyperparameters: Optional[Dict[str, Any]] = None
    split: SplitQueryParams = Field(default_factory=default_training_split)
    study: Optional[FilterValue] = None
    dataset: Optional[FilterValue] = None
    cell_line: Optional[FilterValue] = None
    pe_system: Optional[FilterValue] = None
    edit_type: Optional[FilterValue] = None
    edit_length: Optional[FilterValue] = None
    edit_efficiency_min: Optional[float] = None
    edit_efficiency_max: Optional[float] = None
    edit_scope: Optional[FilterValue] = None
    experimental_method: Optional[FilterValue] = None
    target_context: Optional[FilterValue] = None
    scaffold_name: Optional[FilterValue] = None
    records: Optional[List[Dict[str, Any]]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    device: Optional[str] = "auto"


class TrainingJobSummary(BaseModel):
    job_id: str
    status: JobStatus
    model_name: str
    dataset_name: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    device_requested: Optional[str] = None
    device_assigned: Optional[str] = None
    queue_position: Optional[int] = None
    weights_id: Optional[str] = None
    weights_label: Optional[str] = None
    error: Optional[str] = None


class TrainingJobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class TrainingLogResponse(BaseModel):
    job_id: str
    status: JobStatus
    offset: int
    next_offset: int
    log: str

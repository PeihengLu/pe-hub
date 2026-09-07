"""Shared training request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]
HyperparameterMode = Literal["merge", "replace"]
FilterScalar = Union[str, int]
FilterValue = Union[FilterScalar, List[FilterScalar]]
JobStatus = Literal["queued", "running", "stopping", "succeeded", "failed", "cancelled", "skipped"]


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

    @model_validator(mode="after")
    def _validate_strategy_fields(self) -> "SplitQueryParams":
        """Reject inconsistent split requests before any data is fetched.

        ``pe_common.splits.SplitConfig`` performs the same checks, but only once
        PE-DB has already loaded and converted the data. Validating here turns a
        typo into an immediate 422 instead of a queued job that fails minutes
        later.
        """
        required = {
            "holdout_2": ("train_pct", "test_pct"),
            "holdout_3": ("train_pct", "val_pct", "test_pct"),
        }.get(self.split_strategy)

        if required:
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"split_strategy={self.split_strategy!r} requires {', '.join(missing)}"
                )
            fractions = [float(getattr(self, name)) for name in required]
            if any(value <= 0 or value >= 1 for value in fractions):
                raise ValueError(
                    f"split_strategy={self.split_strategy!r} fractions must each be "
                    "strictly between 0 and 1"
                )
            total = sum(fractions)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"split_strategy={self.split_strategy!r} fractions must sum to 1.0 "
                    f"(got {total:.6g})"
                )

        if self.split_strategy == "cv":
            if self.cv_folds is None:
                raise ValueError("split_strategy='cv' requires cv_folds")
            if int(self.cv_folds) < 2:
                raise ValueError("cv_folds must be >= 2")
            if self.test_pct is not None and not 0 < float(self.test_pct) < 1:
                raise ValueError("test_pct must be strictly between 0 and 1")

        return self


class TrainingRequest(BaseModel):
    model_name: str
    dataset_source: str
    dataset_name: str
    hyperparameters: Optional[Dict[str, Any]] = None
    hyperparameter_mode: HyperparameterMode = "merge"
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

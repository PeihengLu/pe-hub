"""Shared evaluation request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from ..training.schemas import FilterValue, JobStatus, SplitQueryParams

EvaluationJobStatus = JobStatus


def default_evaluation_split() -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_2",
        train_pct=0.8,
        test_pct=0.2,
        use_original_fold=True,
        original_fold_test_value=-1.0,
    )


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    model_name: str
    benchmark_name: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable benchmark label. When omitted and "
            "auto_training_benchmark is enabled, derived from the weight set's "
            "recorded training metadata."
        ),
    )
    weights: str = Field(..., min_length=1, description="Registered weight set ID")
    split: SplitQueryParams = Field(default_factory=default_evaluation_split)
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
    device: Optional[str] = "auto"
    auto_training_benchmark: bool = Field(
        default=True,
        description=(
            "When True (default), evaluation filters and split are taken from the "
            "weight set's recorded training metadata so the held-out test "
            "partition matches training. Set False to supply a custom benchmark."
        ),
    )
    allow_data_leak: bool = Field(
        default=False,
        description=(
            "When False (default), evaluation aborts with a parseable data_leak "
            "error result if the test split overlaps the model's recorded "
            "training loci or leak cannot be ruled out (e.g. no original test "
            "split). Set True to proceed anyway and attach a leak warning."
        ),
    )


class EvaluationJobSummary(BaseModel):
    job_id: str
    status: EvaluationJobStatus
    model_name: str
    benchmark_name: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    device_requested: Optional[str] = None
    device_assigned: Optional[str] = None
    queue_position: Optional[int] = None
    weights_id: Optional[str] = None
    error: Optional[str] = None


class EvaluationJobCreatedResponse(BaseModel):
    job_id: str
    status: EvaluationJobStatus
    message: str


class EvaluationLogResponse(BaseModel):
    job_id: str
    status: EvaluationJobStatus
    offset: int
    next_offset: int
    log: str

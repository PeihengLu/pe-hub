"""Shared evaluation request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pe_common.filter_params import CatalogFilterBody

from ..training.schemas import JobStatus, SplitQueryParams

EvaluationJobStatus = JobStatus


def default_evaluation_split() -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_2",
        train_pct=0.8,
        test_pct=0.2,
        use_original_fold=True,
        original_fold_test_value=-1.0,
    )


class EvaluationRequest(CatalogFilterBody):
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
            "When False (default), in-domain eval aborts with "
            "no_original_test_split when the test is synthetic or the weight "
            "set's has_original_test_split is false (OptiPrime pooled CV is "
            "not Yu/Mathis holdouts). Author-holdout locus overlap on a weight "
            "that used that holdout excludes overlapping target loci and "
            "continues when any remain; full overlap or unverifiable provenance "
            "still aborts. Set True to keep overlapping rows and attach a leak "
            "warning instead."
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

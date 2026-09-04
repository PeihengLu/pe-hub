"""Shared ensemble request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..evaluation.schemas import default_evaluation_split
from ..training.schemas import FilterValue, JobStatus, SplitQueryParams
from .combine import COMBINE_METHODS, CombineMethod

EnsembleJobStatus = JobStatus


class EnsembleMember(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    model_name: str
    weights: str = Field(..., min_length=1, description="Registered weight set ID")
    member_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional weight for weighted_mean (normalized server-side)",
    )


class EnsembleRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ensemble_name: str = Field(..., min_length=1)
    combine: CombineMethod = "mean"
    combine_options: Dict[str, Any] = Field(default_factory=dict)
    members: List[EnsembleMember] = Field(..., min_length=2)
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
    allow_data_leak: bool = Field(
        default=False,
        description=(
            "When False (default), overlapping target loci from the union of "
            "member training provenance are excluded from the test partition. "
            "Full overlap or unverifiable provenance still aborts with a "
            "parseable data_leak error. Set True to keep overlapping rows "
            "and only emit leak_warning."
        ),
    )

    @field_validator("combine")
    @classmethod
    def _validate_combine(cls, value: str) -> str:
        if value not in COMBINE_METHODS:
            raise ValueError(f"combine must be one of: {', '.join(COMBINE_METHODS)}")
        return value

    @model_validator(mode="after")
    def _validate_members(self) -> "EnsembleRequest":
        if len(self.members) < 2:
            raise ValueError("At least two ensemble members are required")
        if self.combine == "weighted_mean":
            explicit = [member.member_weight for member in self.members]
            if any(weight is None for weight in explicit):
                if "weights" not in self.combine_options:
                    raise ValueError(
                        "weighted_mean requires member_weight on each member "
                        "or a weights list in combine_options"
                    )
        return self


class EnsembleJobSummary(BaseModel):
    job_id: str
    status: EnsembleJobStatus
    ensemble_name: str
    combine: CombineMethod
    member_count: int
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    device_requested: Optional[str] = None
    device_assigned: Optional[str] = None
    queue_position: Optional[int] = None
    error: Optional[str] = None


class EnsembleJobCreatedResponse(BaseModel):
    job_id: str
    status: EnsembleJobStatus
    message: str


class EnsembleLogResponse(BaseModel):
    job_id: str
    status: EnsembleJobStatus
    offset: int
    next_offset: int
    log: str

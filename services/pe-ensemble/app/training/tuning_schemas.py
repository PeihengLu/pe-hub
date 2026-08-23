"""Hyperparameter tuning request/response schemas."""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schemas import JobStatus, TrainingRequest

TuningJobStatus = JobStatus


class TuningRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    training: TrainingRequest
    n_trials: int = Field(default=20, ge=1)
    study_name: Optional[str] = None
    study_storage: Optional[str] = None
    write_preset: Optional[str] = None
    no_write_preset: bool = False
    register_best_weights: bool = False


class TuningJobSummary(BaseModel):
    job_id: str
    status: TuningJobStatus
    model_name: str
    dataset_name: str
    n_trials: int
    study_name: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    device_requested: Optional[str] = None
    device_assigned: Optional[str] = None
    queue_position: Optional[int] = None
    best_trial: Optional[int] = None
    best_value: Optional[float] = None
    preset_path: Optional[str] = None
    error: Optional[str] = None


class TuningJobCreatedResponse(BaseModel):
    job_id: str
    status: TuningJobStatus
    message: str


class TuningLogResponse(BaseModel):
    job_id: str
    status: TuningJobStatus
    offset: int
    next_offset: int
    log: str

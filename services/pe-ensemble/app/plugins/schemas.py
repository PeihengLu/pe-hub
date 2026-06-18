"""Plugin validation API schemas."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel

from ..training.schemas import JobStatus


class PluginValidationJobCreatedResponse(BaseModel):
    job_id: str
    plugin_name: str
    status: JobStatus
    message: str


class PluginValidationJobSummary(BaseModel):
    job_id: str
    plugin_name: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class PluginValidationLogResponse(BaseModel):
    job_id: str
    plugin_name: str
    status: JobStatus
    offset: int
    next_offset: int
    log: str


def job_summary(manifest: Dict[str, Any]) -> PluginValidationJobSummary:
    return PluginValidationJobSummary(
        job_id=manifest["job_id"],
        plugin_name=manifest["plugin_name"],
        status=manifest["status"],
        created_at=manifest["created_at"],
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        error=manifest.get("error"),
    )

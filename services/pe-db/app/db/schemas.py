"""Pydantic schemas for catalog API responses."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ScaffoldRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    sequence: str
    description: Optional[str] = None


class StudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    publication_date: Optional[date] = None
    authors: Optional[str] = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str] = None
    assay_type: Optional[str] = None
    study_id: int
    study_name: Optional[str] = None


class DatasheetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    dataset_id: int
    cell_line: str
    pe_system: str
    scaffold_id: str
    num_samples: int
    updated_at: Optional[datetime] = None
    study_name: Optional[str] = None
    dataset_name: Optional[str] = None
    scaffold: Optional[ScaffoldRead] = None

"""Pydantic schemas for catalog API responses."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ScaffoldRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
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
    pegRNA_delivery_method: Optional[str] = None
    pe_delivery_method: Optional[str] = None
    edit_scope: Optional[str] = None
    experimental_method: Optional[str] = None
    target_context: Optional[str] = None
    standardizable: bool = True
    study_id: int
    study_name: Optional[str] = None


class DatasheetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    dataset_id: int
    cell_line: str
    pe_system: str
    scaffold_id: int
    num_samples: int
    updated_at: Optional[datetime] = None
    study_name: Optional[str] = None
    dataset_name: Optional[str] = None
    scaffold: Optional[ScaffoldRead] = None


class StatRow(BaseModel):
    study: str
    count: int


class EditTypeStatRow(StatRow):
    edit_type: str


class EditLengthStatRow(StatRow):
    edit_length: int


class DeliveryMethodStatRow(StatRow):
    delivery_method: str


class EditScopeStatRow(StatRow):
    edit_scope: str


class ExperimentalMethodStatRow(StatRow):
    experimental_method: str


class TargetContextStatRow(StatRow):
    target_context: str


class StatisticsRead(BaseModel):
    edit_type: list[EditTypeStatRow]
    edit_length: list[EditLengthStatRow]
    pegRNA_delivery_method: list[DeliveryMethodStatRow]
    pe_delivery_method: list[DeliveryMethodStatRow]
    edit_scope: list[EditScopeStatRow]
    experimental_method: list[ExperimentalMethodStatRow]
    target_context: list[TargetContextStatRow]
    total_entries: int
    total_studies: int

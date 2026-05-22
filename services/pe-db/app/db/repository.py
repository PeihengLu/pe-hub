"""Read/query helpers for the PE Database catalog."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .models import Dataset, Datasheet, Scaffold, Study
from .schemas import DatasetRead, DatasheetRead, ScaffoldRead, StudyRead


class CatalogRepository:
    def __init__(self, session: Session):
        self._session = session

    def list_studies(self) -> list[StudyRead]:
        rows = self._session.scalars(select(Study).order_by(Study.name)).all()
        return [StudyRead.model_validate(row) for row in rows]

    def get_study_by_name(self, name: str) -> Optional[Study]:
        normalized = name.strip().lower()
        return self._session.scalar(
            select(Study).where(Study.name == normalized)
        )

    def list_scaffolds(self) -> list[ScaffoldRead]:
        rows = self._session.scalars(select(Scaffold).order_by(Scaffold.id)).all()
        return [ScaffoldRead.model_validate(row) for row in rows]

    def get_scaffold(self, scaffold_id: int) -> Optional[ScaffoldRead]:
        row = self._session.get(Scaffold, scaffold_id)
        return ScaffoldRead.model_validate(row) if row else None

    def list_datasets(self, *, study_name: Optional[str] = None) -> list[DatasetRead]:
        stmt = select(Dataset).join(Study).order_by(Study.name, Dataset.name)
        if study_name:
            stmt = stmt.where(Study.name == study_name.strip().lower())
        rows = self._session.scalars(stmt).all()
        return [
            DatasetRead(
                id=row.id,
                name=row.name,
                description=row.description,
                pegRNA_delivery_method=row.pegRNA_delivery_method,
                pe_delivery_method=row.pe_delivery_method,
                edit_scope=row.edit_scope,
                experimental_method=row.experimental_method,
                target_context=row.target_context,
                standardizable=row.standardizable,
                study_id=row.study_id,
                study_name=row.study.name if row.study else None,
            )
            for row in rows
        ]

    def list_datasheets(
        self,
        *,
        study_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> list[DatasheetRead]:
        stmt = (
            select(Datasheet)
            .join(Dataset)
            .join(Study)
            .options(joinedload(Datasheet.scaffold), joinedload(Datasheet.dataset).joinedload(Dataset.study))
            .order_by(Study.name, Dataset.name, Datasheet.cell_line, Datasheet.pe_system)
        )
        if study_name:
            stmt = stmt.where(Study.name == study_name.strip().lower())
        if dataset_name:
            stmt = stmt.where(Dataset.name == dataset_name.strip().lower())
        rows = self._session.scalars(stmt).unique().all()
        return [self._datasheet_to_read(row) for row in rows]

    def find_datasheet(
        self,
        *,
        study_name: str,
        dataset_name: str,
        cell_line: str,
        pe_system: str,
    ) -> Optional[DatasheetRead]:
        stmt = (
            select(Datasheet)
            .join(Dataset)
            .join(Study)
            .options(joinedload(Datasheet.scaffold), joinedload(Datasheet.dataset).joinedload(Dataset.study))
            .where(
                Study.name == study_name.strip().lower(),
                Dataset.name == dataset_name.strip().lower(),
                Datasheet.cell_line == cell_line.strip().lower().replace("-", "_"),
                Datasheet.pe_system == pe_system.strip().lower().replace("-", "_"),
            )
        )
        row = self._session.scalar(stmt)
        return self._datasheet_to_read(row) if row else None

    @staticmethod
    def _datasheet_to_read(row: Datasheet) -> DatasheetRead:
        return DatasheetRead(
            id=row.id,
            file_path=row.file_path,
            dataset_id=row.dataset_id,
            cell_line=row.cell_line,
            pe_system=row.pe_system,
            scaffold_id=row.scaffold_id,
            num_samples=row.num_samples,
            updated_at=row.updated_at,
            study_name=row.dataset.study.name if row.dataset and row.dataset.study else None,
            dataset_name=row.dataset.name if row.dataset else None,
            scaffold=ScaffoldRead.model_validate(row.scaffold) if row.scaffold else None,
        )

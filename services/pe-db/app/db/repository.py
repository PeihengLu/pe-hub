"""Read/query helpers for the PE Database catalog."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from ..config import get_settings
from ..loaders import PEDataLoader
from .models import Dataset, Datasheet, Scaffold, Study
from .schemas import DatasetRead, DatasheetRead, ScaffoldRead, StudyRead

_EDIT_TYPE_CODES: dict[str, int] = {"sub": 0, "ins": 1, "del": 2}


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

    def filter_datasheets(
        self,
        *,
        edit_scope: Optional[str] = None,
        experimental_method: Optional[str] = None,
        target_context: Optional[str] = None,
        scaffold_name: Optional[str] = None,
    ) -> list[DatasheetRead]:
        """Return datasheets whose parent dataset matches catalog metadata filters."""
        rows = self._session.scalars(
            self._apply_catalog_filters(
                self._datasheet_base_stmt(),
                edit_scope=edit_scope,
                experimental_method=experimental_method,
                target_context=target_context,
                scaffold_name=scaffold_name,
            )
        ).unique().all()
        return [self._datasheet_to_read(row) for row in rows]

    def filter_entries(
        self,
        *,
        edit_type: Optional[str] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> list[DatasheetRead]:
        """Return datasheets that contain at least one matching edit row on disk."""
        rows = self._session.scalars(self._datasheet_base_stmt()).unique().all()
        return self._filter_rows_by_entries(
            rows,
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
        )

    def filter_all(
        self,
        *,
        edit_type: Optional[str] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
        edit_scope: Optional[str] = None,
        experimental_method: Optional[str] = None,
        target_context: Optional[str] = None,
        scaffold_name: Optional[str] = None,
    ) -> list[DatasheetRead]:
        """Apply catalog metadata filters, then per-edit filters on standardized data."""
        rows = self._session.scalars(
            self._apply_catalog_filters(
                self._datasheet_base_stmt(),
                edit_scope=edit_scope,
                experimental_method=experimental_method,
                target_context=target_context,
                scaffold_name=scaffold_name,
            )
        ).unique().all()
        return self._filter_rows_by_entries(
            rows,
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
        )

    def _datasheet_base_stmt(self) -> Select[tuple[Datasheet]]:
        return (
            select(Datasheet)
            .join(Dataset)
            .join(Study)
            .options(
                joinedload(Datasheet.scaffold),
                joinedload(Datasheet.dataset).joinedload(Dataset.study),
            )
            .order_by(Study.name, Dataset.name, Datasheet.cell_line, Datasheet.pe_system)
        )

    def _apply_catalog_filters(
        self,
        stmt: Select[tuple[Datasheet]],
        *,
        edit_scope: Optional[str] = None,
        experimental_method: Optional[str] = None,
        target_context: Optional[str] = None,
        scaffold_name: Optional[str] = None,
    ) -> Select[tuple[Datasheet]]:
        if edit_scope is not None:
            stmt = stmt.where(Dataset.edit_scope == edit_scope.strip().lower())
        if experimental_method is not None:
            stmt = stmt.where(
                Dataset.experimental_method == experimental_method.strip().lower()
            )
        if target_context is not None:
            stmt = stmt.where(Dataset.target_context == target_context.strip().lower())
        if scaffold_name is not None:
            normalized = scaffold_name.strip().lower()
            allowed = self._allowed_scaffold_names()
            if normalized not in allowed:
                raise ValueError(
                    "Unknown scaffold_name "
                    f"{scaffold_name!r}; expected one of: {', '.join(sorted(allowed))}"
                )
            stmt = stmt.join(Scaffold).where(func.lower(Scaffold.name) == normalized)
        return stmt

    def _filter_rows_by_entries(
        self,
        rows: list[Datasheet],
        *,
        edit_type: Optional[str] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> list[DatasheetRead]:
        if not self._entry_filters_active(
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
        ):
            return [self._datasheet_to_read(row) for row in rows]

        loader = PEDataLoader(get_settings().data_root)
        edit_type_code = self._parse_edit_type(edit_type) if edit_type is not None else None
        matched: list[DatasheetRead] = []
        for row in rows:
            if row.dataset is None or row.dataset.study is None:
                continue
            data = self._load_standardized_for_datasheet(row, loader)
            if data is None:
                continue
            if self._dataframe_has_matching_entries(
                data,
                edit_type_code=edit_type_code,
                edit_length=edit_length,
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            ):
                matched.append(self._datasheet_to_read(row))
        return matched

    @staticmethod
    def _entry_filters_active(
        *,
        edit_type: Optional[str] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> bool:
        return any(
            value is not None
            for value in (
                edit_type,
                edit_length,
                edit_efficiency_min,
                edit_efficiency_max,
            )
        )

    @staticmethod
    def _parse_edit_type(edit_type: str) -> int:
        key = edit_type.strip().lower()
        if key not in _EDIT_TYPE_CODES:
            raise ValueError(
                f"Unknown edit_type {edit_type!r}; expected one of: sub, ins, del"
            )
        return _EDIT_TYPE_CODES[key]

    def _allowed_scaffold_names(self) -> set[str]:
        names = self._session.scalars(select(func.lower(Scaffold.name))).all()
        return {name for name in names if name}

    @staticmethod
    def _load_standardized_for_datasheet(
        row: Datasheet, loader: PEDataLoader
    ) -> Optional[pd.DataFrame]:
        study = row.dataset.study.name
        dataset = row.dataset.name
        path = loader._find_standardized_file(
            study=study,
            dataset=dataset,
            cell_line=row.cell_line,
            pe_system=row.pe_system,
        )
        if not path.exists():
            return None
        return loader._read_dataframe(path)

    @staticmethod
    def _dataframe_has_matching_entries(
        df: pd.DataFrame,
        *,
        edit_type_code: Optional[int] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> bool:
        if df.empty:
            return False

        mask = pd.Series(True, index=df.index)
        if edit_length is not None:
            length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
            if length_col not in df.columns:
                return False
            mask &= pd.to_numeric(df[length_col], errors="coerce") == edit_length

        if edit_type_code is not None:
            type_columns = ("type_sub", "type_ins", "type_del")
            if not set(type_columns).issubset(df.columns):
                return False
            type_masks = {
                0: df["type_sub"].astype(bool),
                1: df["type_ins"].astype(bool),
                2: df["type_del"].astype(bool),
            }
            mask &= type_masks[edit_type_code]

        if edit_efficiency_min is not None or edit_efficiency_max is not None:
            if "editing_efficiency" not in df.columns:
                return False
            efficiency = pd.to_numeric(df["editing_efficiency"], errors="coerce")
            if edit_efficiency_min is not None:
                mask &= efficiency >= edit_efficiency_min
            if edit_efficiency_max is not None:
                mask &= efficiency <= edit_efficiency_max

        return bool(mask.any())

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

"""Read/query helpers for the PE Database catalog."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload

from ..config import get_settings
from ..loaders import PEDataLoader
from .models import Dataset, Datasheet, Scaffold, Study
from .schemas import (
    DatasetRead,
    DatasheetRead,
    DeliveryMethodStatRow,
    EditLengthStatRow,
    EditScopeStatRow,
    EditTypeStatRow,
    ExperimentalMethodStatRow,
    ScaffoldRead,
    StatisticsRead,
    StudyRead,
    TargetContextStatRow,
)

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

    def compute_statistics(
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
    ) -> StatisticsRead:
        """Descriptive statistics over edit rows, optionally narrowed by catalog/entry filters."""
        rows = self._session.scalars(
            self._apply_catalog_filters(
                self._datasheet_base_stmt(),
                edit_scope=edit_scope,
                experimental_method=experimental_method,
                target_context=target_context,
                scaffold_name=scaffold_name,
            )
        ).unique().all()

        loader = PEDataLoader(get_settings().data_root)
        edit_type_code = self._parse_edit_type(edit_type) if edit_type is not None else None

        entry_records: list[dict[str, Any]] = []
        for row in rows:
            if row.dataset is None or row.dataset.study is None:
                continue
            data = self._load_standardized_for_datasheet(row, loader)
            if data is None or data.empty:
                continue
            filtered = self._filter_dataframe_entries(
                data,
                edit_type_code=edit_type_code,
                edit_length=edit_length,
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            )
            if filtered.empty:
                continue

            study_name = row.dataset.study.name
            dataset = row.dataset
            edit_types = self._extract_edit_type_series(filtered)
            edit_lengths = self._extract_edit_length_series(filtered)
            for idx in filtered.index:
                entry_records.append(
                    {
                        "study": study_name,
                        "edit_type": edit_types.loc[idx],
                        "edit_length": edit_lengths.loc[idx],
                        "pegRNA_delivery_method": dataset.pegRNA_delivery_method,
                        "pe_delivery_method": dataset.pe_delivery_method,
                        "edit_scope": dataset.edit_scope,
                        "experimental_method": dataset.experimental_method,
                        "target_context": dataset.target_context,
                    }
                )

        all_studies = {record["study"] for record in entry_records}
        return StatisticsRead(
            edit_type=self._build_stat_rows(
                entry_records, "edit_type", EditTypeStatRow, "edit_type"
            ),
            edit_length=self._build_stat_rows(
                entry_records, "edit_length", EditLengthStatRow, "edit_length"
            ),
            pegRNA_delivery_method=self._build_stat_rows(
                entry_records,
                "pegRNA_delivery_method",
                DeliveryMethodStatRow,
                "delivery_method",
            ),
            pe_delivery_method=self._build_stat_rows(
                entry_records,
                "pe_delivery_method",
                DeliveryMethodStatRow,
                "delivery_method",
            ),
            edit_scope=self._build_stat_rows(
                entry_records, "edit_scope", EditScopeStatRow, "edit_scope"
            ),
            experimental_method=self._build_stat_rows(
                entry_records,
                "experimental_method",
                ExperimentalMethodStatRow,
                "experimental_method",
            ),
            target_context=self._build_stat_rows(
                entry_records, "target_context", TargetContextStatRow, "target_context"
            ),
            total_entries=len(entry_records),
            total_studies=len(all_studies),
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
            filtered = self._filter_dataframe_entries(
                data,
                edit_type_code=edit_type_code,
                edit_length=edit_length,
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            )
            if not filtered.empty:
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
    def _filter_dataframe_entries(
        df: pd.DataFrame,
        *,
        edit_type_code: Optional[int] = None,
        edit_length: Optional[int] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        mask = pd.Series(True, index=df.index)
        if edit_length is not None:
            length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
            if length_col not in df.columns:
                return df.iloc[0:0]
            mask &= pd.to_numeric(df[length_col], errors="coerce") == edit_length

        if edit_type_code is not None:
            type_columns = ("type_sub", "type_ins", "type_del")
            if not set(type_columns).issubset(df.columns):
                return df.iloc[0:0]
            type_masks = {
                0: df["type_sub"].astype(bool),
                1: df["type_ins"].astype(bool),
                2: df["type_del"].astype(bool),
            }
            mask &= type_masks[edit_type_code]

        if edit_efficiency_min is not None or edit_efficiency_max is not None:
            if "editing_efficiency" not in df.columns:
                return df.iloc[0:0]
            efficiency = pd.to_numeric(df["editing_efficiency"], errors="coerce")
            if edit_efficiency_min is not None:
                mask &= efficiency >= edit_efficiency_min
            if edit_efficiency_max is not None:
                mask &= efficiency <= edit_efficiency_max

        return df.loc[mask]

    @staticmethod
    def _extract_edit_type_series(df: pd.DataFrame) -> pd.Series:
        type_columns = ("type_sub", "type_ins", "type_del")
        if not set(type_columns).issubset(df.columns):
            return pd.Series([None] * len(df), index=df.index, dtype=object)
        return pd.Series(
            np.select(
                [
                    df["type_sub"].astype(bool),
                    df["type_ins"].astype(bool),
                    df["type_del"].astype(bool),
                ],
                ["sub", "ins", "del"],
                default=None,
            ),
            index=df.index,
            dtype=object,
        )

    @staticmethod
    def _extract_edit_length_series(df: pd.DataFrame) -> pd.Series:
        length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
        if length_col not in df.columns:
            return pd.Series([None] * len(df), index=df.index, dtype=object)
        return pd.to_numeric(df[length_col], errors="coerce")

    @staticmethod
    def _build_stat_rows(
        records: list[dict[str, Any]],
        field: str,
        row_model: type,
        label_field: str,
    ) -> list:
        buckets: dict[tuple[Any, str], int] = defaultdict(int)
        for record in records:
            value = record.get(field)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            buckets[(value, record["study"])] += 1

        rows = []
        for (value, study) in sorted(
            buckets,
            key=lambda item: (str(type(item[0])), item[0], item[1]),
        ):
            normalized_value = int(value) if label_field == "edit_length" else value
            rows.append(
                row_model(
                    **{
                        label_field: normalized_value,
                        "study": study,
                        "count": buckets[(value, study)],
                    }
                )
            )
        return rows

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

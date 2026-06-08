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
from ..utils.json_utils import dataframe_to_json_records
from pe_common.splits import SplitConfig, assign_splits
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
_SPLIT_GROUP_SCOPE_COL = "_split_group_scope"


def _datasheet_scope_key(descriptor: dict[str, Any]) -> str:
    return (
        f"{descriptor['study']}|{descriptor['dataset']}|"
        f"{descriptor['cell_line']}|{descriptor['pe_system']}"
    )


def _attach_split_metadata(converted: pd.DataFrame, standardized: pd.DataFrame) -> pd.DataFrame:
    output = converted.copy()
    for column in ("split", "split_source"):
        if column in standardized.columns:
            output[column] = standardized[column].to_numpy()
    if "original_fold" in standardized.columns:
        output["original_fold"] = standardized["original_fold"].to_numpy()
    return output


def _apply_export_split(
    standardized: pd.DataFrame,
    split_config: SplitConfig,
    *,
    merge_groups: bool,
    group_scope: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split_config.strategy == "none":
        return standardized, {"strategy": "none"}

    if merge_groups:
        scoped = standardized.copy()
        if _SPLIT_GROUP_SCOPE_COL not in scoped.columns:
            scoped[_SPLIT_GROUP_SCOPE_COL] = group_scope or "merged"
        merged_config = SplitConfig(
            strategy=split_config.strategy,
            train_pct=split_config.train_pct,
            val_pct=split_config.val_pct,
            test_pct=split_config.test_pct,
            cv_folds=split_config.cv_folds,
            use_original_fold=split_config.use_original_fold,
            original_fold_test_value=split_config.original_fold_test_value,
            original_fold_col=split_config.original_fold_col,
            group_col=split_config.group_col,
            random_state=split_config.random_state,
            fold_namespace_prefix=split_config.fold_namespace_prefix,
            group_scope_col=_SPLIT_GROUP_SCOPE_COL,
        )
        split_df, summary = assign_splits(scoped, merged_config)
        return split_df.drop(columns=[_SPLIT_GROUP_SCOPE_COL], errors="ignore"), summary

    return assign_splits(standardized, split_config)


def _normalize_filter_values(values: Optional[str | list[str]]) -> Optional[list[str]]:
    """Collapse absent, scalar, or repeated filter params into a deduped list."""
    if values is None:
        return None
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw:
        token = value.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(token)
    return normalized or None


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
        study_name: Optional[str | list[str]] = None,
        dataset_name: Optional[str | list[str]] = None,
        cell_line: Optional[str | list[str]] = None,
        pe_system: Optional[str | list[str]] = None,
        edit_type: Optional[str | list[str]] = None,
        edit_length: Optional[int | list[int]] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
        edit_scope: Optional[str | list[str]] = None,
        experimental_method: Optional[str | list[str]] = None,
        target_context: Optional[str | list[str]] = None,
        scaffold_name: Optional[str | list[str]] = None,
        target_format: Optional[str] = None,
        split_config: Optional[SplitConfig] = None,
        merge_groups: bool = False,
    ) -> list[DatasheetRead] | dict[str, Any]:
        """Apply catalog metadata filters, then per-edit filters on standardized data.

        Args:
            study_name / dataset_name: Narrow to a study and/or dataset.
            cell_line / pe_system: Narrow to specific datasheets (e.g. to target a
                single datasheet for conversion, like the point-lookup data API).
            edit_* / *_method / *_scope / *_context / scaffold_name: Catalog and
                per-edit filters (unchanged behavior).
            target_format: When ``None``, return the matching datasheets as usual.
                When set (``std`` or a model format such as ``deepprime``,
                ``pridict``, ``pridict2``, ``oped``), return the matching rows
                converted from standardized data into the requested format,
                including only datasets flagged ``standardizable`` in the catalog.
                This is the single entry point for standardized -> model-format
                conversion (single or bulk).

        Returns:
            ``list[DatasheetRead]`` when ``target_format`` is ``None``; otherwise a
            dict with converted ``groups`` (one per datasheet), ``skipped`` entries
            (non-standardizable or unavailable), and ``total_records``.
        """
        rows = self._session.scalars(
            self._apply_catalog_filters(
                self._datasheet_base_stmt(),
                study_name=study_name,
                dataset_name=dataset_name,
                cell_line=cell_line,
                pe_system=pe_system,
                edit_scope=edit_scope,
                experimental_method=experimental_method,
                target_context=target_context,
                scaffold_name=scaffold_name,
            )
        ).unique().all()

        if target_format is None:
            return self._filter_rows_by_entries(
                rows,
                edit_type=edit_type,
                edit_length=edit_length,
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            )

        return self._convert_filtered_rows_to_format(
            rows,
            target_format=target_format,
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
            split_config=split_config,
            merge_groups=merge_groups,
        )

    def _convert_filtered_rows_to_format(
        self,
        rows: list[Datasheet],
        *,
        target_format: str,
        edit_type: Optional[str | list[str]] = None,
        edit_length: Optional[int | list[int]] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
        split_config: Optional[SplitConfig] = None,
        merge_groups: bool = False,
    ) -> dict[str, Any]:
        """Convert standardizable datasheets' standardized data into ``target_format``."""
        from ..converter import DataConverter

        data_root = get_settings().data_root
        loader = PEDataLoader(data_root)
        converter = DataConverter(data_root)
        edit_type_codes = self._parse_edit_types(edit_type)

        split_config = split_config or SplitConfig(strategy="none")
        groups: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        pending: list[tuple[dict[str, Any], pd.DataFrame]] = []
        split_summaries: list[dict[str, Any]] = []

        for row in rows:
            if row.dataset is None or row.dataset.study is None:
                continue
            dataset = row.dataset
            descriptor = {
                "study": dataset.study.name,
                "dataset": dataset.name,
                "cell_line": row.cell_line,
                "pe_system": row.pe_system,
            }

            if not dataset.standardizable:
                skipped.append({**descriptor, "reason": "dataset not standardizable"})
                continue

            data = self._load_standardized_for_datasheet(row, loader)
            if data is None or data.empty:
                skipped.append({**descriptor, "reason": "no standardized data on disk"})
                continue

            filtered = self._filter_dataframe_entries(
                data,
                edit_type_codes=edit_type_codes,
                edit_lengths=self._normalize_edit_lengths(edit_length),
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            )
            if filtered.empty:
                continue

            pending.append((descriptor, filtered))

        if merge_groups and pending:
            merged_frames: list[pd.DataFrame] = []
            merged_descriptor = {
                "study": "merged",
                "dataset": "merged",
                "cell_line": "merged",
                "pe_system": "merged",
            }
            for descriptor, filtered in pending:
                scoped = filtered.copy()
                scoped[_SPLIT_GROUP_SCOPE_COL] = _datasheet_scope_key(descriptor)
                merged_frames.append(scoped)
            merged_std = pd.concat(merged_frames, ignore_index=True)
            try:
                split_std, split_summary = _apply_export_split(
                    merged_std,
                    split_config,
                    merge_groups=True,
                )
                converted = converter.convert_from_standardized(
                    split_std,
                    study="merged",
                    dataset="merged",
                    cell_line="merged",
                    pe_system="merged",
                    target_format=target_format,
                )
                converted = _attach_split_metadata(converted, split_std)
            except ValueError as exc:
                skipped.append({**merged_descriptor, "reason": f"split assignment failed: {exc}"})
            else:
                groups.append(
                    {
                        **merged_descriptor,
                        "num_records": int(len(converted)),
                        "columns": list(converted.columns),
                        "records": dataframe_to_json_records(converted),
                    }
                )
                split_summaries.append(split_summary)
        else:
            for descriptor, filtered in pending:
                try:
                    split_std, split_summary = _apply_export_split(
                        filtered,
                        split_config,
                        merge_groups=False,
                    )
                    converted = converter.convert_from_standardized(
                        split_std,
                        study=descriptor["study"],
                        dataset=descriptor["dataset"],
                        cell_line=descriptor["cell_line"],
                        pe_system=descriptor["pe_system"],
                        target_format=target_format,
                    )
                    converted = _attach_split_metadata(converted, split_std)
                except (ValueError, KeyError) as exc:
                    skipped.append({**descriptor, "reason": str(exc)})
                    continue

                groups.append(
                    {
                        **descriptor,
                        "num_records": int(len(converted)),
                        "columns": list(converted.columns),
                        "records": dataframe_to_json_records(converted),
                    }
                )
                if split_config.strategy != "none":
                    split_summaries.append({**descriptor, **split_summary})

        payload: dict[str, Any] = {
            "target_format": target_format,
            "groups": groups,
            "skipped": skipped,
            "total_records": int(sum(group["num_records"] for group in groups)),
            "merged": merge_groups,
        }
        if split_config.strategy != "none":
            payload["split"] = {
                "strategy": split_config.strategy,
                "use_original_fold": split_config.use_original_fold,
                "random_state": split_config.random_state,
                "summaries": split_summaries,
            }
        return payload

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
        edit_type_codes = self._parse_edit_types(edit_type)
        edit_lengths = self._normalize_edit_lengths(edit_length)

        entry_records: list[dict[str, Any]] = []
        for row in rows:
            if row.dataset is None or row.dataset.study is None:
                continue
            data = self._load_standardized_for_datasheet(row, loader)
            if data is None or data.empty:
                continue
            filtered = self._filter_dataframe_entries(
                data,
                edit_type_codes=edit_type_codes,
                edit_lengths=edit_lengths,
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
        study_name: Optional[str | list[str]] = None,
        dataset_name: Optional[str | list[str]] = None,
        cell_line: Optional[str | list[str]] = None,
        pe_system: Optional[str | list[str]] = None,
        edit_scope: Optional[str | list[str]] = None,
        experimental_method: Optional[str | list[str]] = None,
        target_context: Optional[str | list[str]] = None,
        scaffold_name: Optional[str | list[str]] = None,
    ) -> Select[tuple[Datasheet]]:
        studies = _normalize_filter_values(study_name)
        if studies is not None:
            stmt = stmt.where(Study.name.in_([value.lower() for value in studies]))
        datasets = _normalize_filter_values(dataset_name)
        if datasets is not None:
            stmt = stmt.where(Dataset.name.in_([value.lower() for value in datasets]))
        cell_lines = _normalize_filter_values(cell_line)
        if cell_lines is not None:
            stmt = stmt.where(
                Datasheet.cell_line.in_(
                    [value.lower().replace("-", "_") for value in cell_lines]
                )
            )
        pe_systems = _normalize_filter_values(pe_system)
        if pe_systems is not None:
            stmt = stmt.where(
                Datasheet.pe_system.in_(
                    [value.lower().replace("-", "_") for value in pe_systems]
                )
            )
        scopes = _normalize_filter_values(edit_scope)
        if scopes is not None:
            stmt = stmt.where(Dataset.edit_scope.in_([value.lower() for value in scopes]))
        methods = _normalize_filter_values(experimental_method)
        if methods is not None:
            stmt = stmt.where(
                Dataset.experimental_method.in_([value.lower() for value in methods])
            )
        contexts = _normalize_filter_values(target_context)
        if contexts is not None:
            stmt = stmt.where(
                Dataset.target_context.in_([value.lower() for value in contexts])
            )
        scaffolds = _normalize_filter_values(scaffold_name)
        if scaffolds is not None:
            allowed = self._allowed_scaffold_names()
            normalized = [value.lower() for value in scaffolds]
            unknown = sorted({value for value in normalized if value not in allowed})
            if unknown:
                raise ValueError(
                    "Unknown scaffold_name "
                    f"{unknown!r}; expected one of: {', '.join(sorted(allowed))}"
                )
            stmt = stmt.join(Scaffold).where(func.lower(Scaffold.name).in_(normalized))
        return stmt

    def _filter_rows_by_entries(
        self,
        rows: list[Datasheet],
        *,
        edit_type: Optional[str | list[str]] = None,
        edit_length: Optional[int | list[int]] = None,
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
        edit_type_codes = self._parse_edit_types(edit_type)
        edit_lengths = self._normalize_edit_lengths(edit_length)
        matched: list[DatasheetRead] = []
        for row in rows:
            if row.dataset is None or row.dataset.study is None:
                continue
            data = self._load_standardized_for_datasheet(row, loader)
            if data is None:
                continue
            filtered = self._filter_dataframe_entries(
                data,
                edit_type_codes=edit_type_codes,
                edit_lengths=edit_lengths,
                edit_efficiency_min=edit_efficiency_min,
                edit_efficiency_max=edit_efficiency_max,
            )
            if not filtered.empty:
                matched.append(self._datasheet_to_read(row))
        return matched

    @staticmethod
    def _entry_filters_active(
        *,
        edit_type: Optional[str | list[str]] = None,
        edit_length: Optional[int | list[int]] = None,
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

    @classmethod
    def _parse_edit_types(cls, edit_type: Optional[str | list[str]]) -> Optional[list[int]]:
        values = _normalize_filter_values(edit_type)
        if values is None:
            return None
        return [cls._parse_edit_type(value) for value in values]

    @staticmethod
    def _normalize_edit_lengths(
        edit_length: Optional[int | list[int]],
    ) -> Optional[list[int]]:
        if edit_length is None:
            return None
        if isinstance(edit_length, int):
            return [edit_length]
        normalized: list[int] = []
        seen: set[int] = set()
        for value in edit_length:
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized or None

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
        edit_type_codes: Optional[list[int]] = None,
        edit_lengths: Optional[list[int]] = None,
        edit_efficiency_min: Optional[float] = None,
        edit_efficiency_max: Optional[float] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        mask = pd.Series(True, index=df.index)
        if edit_lengths is not None:
            length_col = "edit_len" if "edit_len" in df.columns else "edit_length"
            if length_col not in df.columns:
                return df.iloc[0:0]
            mask &= pd.to_numeric(df[length_col], errors="coerce").isin(edit_lengths)

        if edit_type_codes is not None:
            type_columns = ("type_sub", "type_ins", "type_del")
            if not set(type_columns).issubset(df.columns):
                return df.iloc[0:0]
            type_masks = {
                0: df["type_sub"].astype(bool),
                1: df["type_ins"].astype(bool),
                2: df["type_del"].astype(bool),
            }
            type_mask = pd.Series(False, index=df.index)
            for code in edit_type_codes:
                type_mask |= type_masks[code]
            mask &= type_mask

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

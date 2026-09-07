"""Catalog record types for studies, datasets, and datasheet scaffold assignment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


def canonical_dataset_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@dataclass(frozen=True)
class StudyRecord:
    key: str
    display_name: str
    publication_date: Optional[date]
    authors: str


@dataclass(frozen=True)
class DatasetRecord:
    """Catalog metadata for a dataset within a study.

    Field semantics:
      - ``pegRNA_delivery_method`` / ``pe_delivery_method``: how each component
        reaches cells (often different, e.g. lentiviral pegRNA + transfected PE).
      - ``edit_scope``: intended vs off-target editing readout (on_target / off_target).
      - ``experimental_method``: where editing is measured (in_vitro / in_vivo).
      - ``target_context``: where the edit is measured — ``endogenous`` (native
        chromosomal locus) vs ``non_endogenous`` (synthetic target).
      - ``standardizable``: full conversion to the shared standardized format.
      - ``partial``: filter-only parquet (no model-format export) when True.
    """

    study_key: str
    name: str
    description: str
    pegRNA_delivery_method: str
    pe_delivery_method: str
    edit_scope: str
    experimental_method: str
    target_context: str
    standardizable: bool = True
    partial: bool = False

    def produces_standardized(self) -> bool:
        return bool(self.standardizable or self.partial)


@dataclass(frozen=True)
class DatasheetScaffoldAssignment:
    """Which pegRNA scaffold applies to one exported datasheet."""

    study: str
    dataset: str
    cell_line: str
    pe_system: str
    scaffold_id: int
    scaffold_source: str


def canonical_name(value: str) -> str:
    return canonical_dataset_name(value)

"""Datasheet catalog: scaffold assignment and indexing from ``datasets/exported/``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Dataset, Datasheet
from ..db.session import get_session
from .scaffolds import (
    default_scaffold_for_pridict,
    scaffold_id_for_minsepie_sequence,
    scaffold_id_from_deepprime_label,
)
from .seed import _upsert_studies_and_datasets

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Scaffold assignment per exported datasheet
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasheetScaffoldAssignment:
    """Which pegRNA scaffold applies to one exported datasheet."""

    study: str
    dataset: str
    cell_line: str
    pe_system: str
    scaffold_id: str
    scaffold_source: str


def _normalize_study_key(value: str) -> str:
    return str(value).strip().lower()


def _normalize_dataset_name(value: str) -> str:
    return str(value).strip().lower()


def _normalize_datasheet_field(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def build_deepprime_scaffold_assignments(data_root: Optional[Path] = None) -> list[DatasheetScaffoldAssignment]:
    root = data_root or get_settings().data_root
    original_excel_path = root / "raw" / "deepprime" / "deepprime-org.xlsx"
    if not original_excel_path.exists():
        return []
    catalog = pd.read_excel(original_excel_path, sheet_name="Summary", header=0)
    dataset_renames = {
        "Library-ClinVar": "deepprime-clinvar",
        "Library-Small": "deepprime-small",
        "Library-Off": "deepprime-off",
        "Library-Off(sub-pool)": "deepprime-off-subpool",
    }
    rows: list[DatasheetScaffoldAssignment] = []
    for _, catalog_row in catalog.iterrows():
        dataset = str(catalog_row["Library"])
        dataset = dataset_renames.get(dataset, dataset)
        scaffold_label = str(catalog_row.get("Scaffold", "")).strip()
        if not scaffold_label:
            continue
        rows.append(
            DatasheetScaffoldAssignment(
                study="deepprime",
                dataset=dataset,
                cell_line=_normalize_datasheet_field(str(catalog_row["Cell line"])),
                pe_system=_normalize_datasheet_field(str(catalog_row["PE system"])),
                scaffold_id=scaffold_id_from_deepprime_label(scaffold_label),
                scaffold_source="deepprime_summary",
            )
        )
    return rows


def build_pridict_scaffold_assignments() -> list[DatasheetScaffoldAssignment]:
    scaffold_id = default_scaffold_for_pridict()
    specs = [
        ("pridict1", "library1", "hek293t", "pe2"),
        ("pridict2", "library-diverse", "hek", "pe2"),
        ("pridict2", "library-diverse", "k562", "pe2"),
        ("pridict2", "library-diverse", "k562mlh1dn", "pe2"),
        ("pridict2", "library-diverse-invivo", "adv", "pe2"),
        # Legacy export path if AdV rows were written under library-diverse
        ("pridict2", "library-diverse", "adv", "pe2"),
    ]
    return [
        DatasheetScaffoldAssignment(
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            scaffold_id=scaffold_id,
            scaffold_source="pridict_pegRNA_design_default",
        )
        for study, dataset, cell_line, pe_system in specs
    ]


def build_minsepie_scaffold_assignments(data_root: Optional[Path] = None) -> list[DatasheetScaffoldAssignment]:
    from ..utils.standardize_data import (
        _MINSEPIE_CELL_ABBR_TO_FULL,
        _MINSEPIE_EXPERIMENT_TO_PEGRNA,
        _load_minsepie_pegrna_table,
        _parse_minsepie_experiment,
    )

    root = data_root or get_settings().data_root
    data_path = root / "raw" / "minsepie" / "41587_2023_1678_MOESM5_ESM.tsv"
    if not data_path.exists():
        return []
    data = pd.read_csv(data_path, sep="\t", usecols=["experiment"])
    pegrna_df = _load_minsepie_pegrna_table()
    pegrna_lookup = {
        (str(row["target"]), str(row["purpose"])): row
        for _, row in pegrna_df.iterrows()
    }
    rows: list[DatasheetScaffoldAssignment] = []
    for experiment in data["experiment"].astype(str).unique():
        key = _MINSEPIE_EXPERIMENT_TO_PEGRNA.get(experiment)
        if key is None or key not in pegrna_lookup:
            continue
        pegrna_row = pegrna_lookup[key]
        scaffold_seq = pegrna_row.get("pegrna_scaffold_seq")
        if pd.isna(scaffold_seq):
            continue
        target_variant, cell_abbr, pe_condition = _parse_minsepie_experiment(experiment)
        cell_line = _MINSEPIE_CELL_ABBR_TO_FULL[cell_abbr]
        pe_system = _normalize_datasheet_field(f"{target_variant}_{pe_condition}")
        rows.append(
            DatasheetScaffoldAssignment(
                study="minsepie",
                dataset="library-insert",
                cell_line=cell_line,
                pe_system=pe_system,
                scaffold_id=scaffold_id_for_minsepie_sequence(str(scaffold_seq)),
                scaffold_source="minsepie_st6",
            )
        )
    return rows


def build_scaffold_assignments(data_root: Optional[Path] = None) -> list[DatasheetScaffoldAssignment]:
    """Infer scaffold_id for every known exported datasheet from raw study metadata."""
    return (
        build_deepprime_scaffold_assignments(data_root)
        + build_pridict_scaffold_assignments()
        + build_minsepie_scaffold_assignments(data_root)
    )


def _scaffold_assignment_lookup(
    assignments: list[DatasheetScaffoldAssignment],
) -> dict[tuple[str, str, str, str], str]:
    return {
        (
            row.study,
            _normalize_dataset_name(row.dataset),
            _normalize_datasheet_field(row.cell_line),
            _normalize_datasheet_field(row.pe_system),
        ): row.scaffold_id
        for row in assignments
    }


# ------------------------------------------------------------------------------
# Index exported CSVs into the Datasheet table
# ------------------------------------------------------------------------------

def _count_samples(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    try:
        return len(pd.read_csv(file_path, low_memory=False))
    except Exception as exc:
        logger.warning("Could not count samples in %s: %s", file_path, exc)
        return 0


def _relative_data_path(file_path: Path, data_root: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(file_path)


def _index_exported_datasheets(
    session: Session,
    dataset_index: dict[tuple[str, str], Dataset],
    scaffold_by_datasheet: dict[tuple[str, str, str, str], str],
    *,
    data_root: Path,
    exported_dir: Path,
) -> int:
    """Register one Datasheet row per CSV under ``datasets/exported/``."""
    updated = 0
    if not exported_dir.exists():
        return updated

    for data_file in exported_dir.rglob("*.csv"):
        if not data_file.is_file():
            continue
        rel_parts = data_file.relative_to(exported_dir).parts
        if len(rel_parts) != 3 or "-" not in data_file.stem:
            continue

        study_key, dataset_name, _ = rel_parts
        cell_line, pe_system = data_file.stem.rsplit("-", 1)
        study_key = _normalize_study_key(study_key)
        dataset_name = _normalize_dataset_name(dataset_name)
        cell_line = _normalize_datasheet_field(cell_line)
        pe_system = _normalize_datasheet_field(pe_system)

        dataset = dataset_index.get((study_key, dataset_name))
        if dataset is None:
            logger.debug(
                "Skipping unregistered export %s (study=%s dataset=%s)",
                data_file,
                study_key,
                dataset_name,
            )
            continue

        scaffold_id = scaffold_by_datasheet.get((study_key, dataset_name, cell_line, pe_system))
        if scaffold_id is None:
            logger.warning(
                "No scaffold mapping for %s/%s %s-%s; defaulting to optimized",
                study_key,
                dataset_name,
                cell_line,
                pe_system,
            )
            scaffold_id = "optimized"

        rel_path = _relative_data_path(data_file, data_root)
        num_samples = _count_samples(data_file)

        existing = session.scalar(
            select(Datasheet).where(
                Datasheet.dataset_id == dataset.id,
                Datasheet.cell_line == cell_line,
                Datasheet.pe_system == pe_system,
            )
        )
        if existing is None:
            session.add(
                Datasheet(
                    file_path=rel_path,
                    dataset_id=dataset.id,
                    cell_line=cell_line,
                    pe_system=pe_system,
                    scaffold_id=scaffold_id,
                    num_samples=num_samples,
                )
            )
        else:
            existing.file_path = rel_path
            existing.scaffold_id = scaffold_id
            existing.num_samples = num_samples
        updated += 1
    return updated


def index_exported_datasheets(
    *,
    data_root: Optional[Path] = None,
    scaffold_assignments: Optional[list[DatasheetScaffoldAssignment]] = None,
) -> int:
    """
    Refresh Datasheet catalog rows from ``datasets/exported/`` CSV files.

    Study, Dataset, and Scaffold must already be seeded (see ``init_catalog``).
    """
    settings = get_settings()
    root = data_root or settings.data_root
    exported_dir = root / "exported"

    assignments = (
        scaffold_assignments
        if scaffold_assignments is not None
        else build_scaffold_assignments(root)
    )
    scaffold_by_datasheet = _scaffold_assignment_lookup(assignments)

    with get_session() as session:
        dataset_index = _upsert_studies_and_datasets(session)
        count = _index_exported_datasheets(
            session,
            dataset_index,
            scaffold_by_datasheet,
            data_root=root,
            exported_dir=exported_dir,
        )
    logger.info("Indexed %s exported datasheets into catalog", count)
    return count

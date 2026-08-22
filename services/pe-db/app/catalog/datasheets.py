"""Datasheet catalog: scaffold assignment and indexing from ``datasets/exported/``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from ..config import get_settings
from ..db.models import Dataset, Datasheet
from ..db.session import get_session
from .scaffolds import (
    MINSEPIE_DATASET_SCAFFOLD_ID,
    SCAFFOLD_ID_CONVENTIONAL,
    SCAFFOLD_ID_OPTIMIZED,
    SCAFFOLD_ID_OPTIPRIME_BLPI_FE,
    default_scaffold_for_pridict,
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
    scaffold_id: int
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
        ("pridict1", "library2", "hek293t", "pe2"),
        ("pridict1", "library2", "hek293tmlh1dn", "pe2"),
        ("pridict1", "library2", "u2os", "pe2"),
        ("pridict1", "library2", "u2osmlh1dn", "pe2"),
        ("pridict1", "library2", "u2os", "pemax"),
        ("pridict1", "library2", "u2osmlh1dn", "pemax"),
        ("pridict1", "library2", "k562", "pe2"),
        ("pridict1", "library2", "k562mlh1dn", "pe2"),
        ("pridict1", "library2", "k562", "pemax"),
        ("pridict1", "library2", "k562mlh1dn", "pemax"),
        ("pridict1", "library2-invivo", "liver_gfpplus", "pe2"),
        ("pridict1", "endogenous", "hek293t", "pe2"),
        ("pridict1", "endogenous", "k562", "pe2"),
        ("pridict2", "library-diverse", "hek", "pe2"),
        ("pridict2", "library-diverse", "k562", "pe2"),
        ("pridict2", "library-diverse", "k562mlh1dn", "pe2"),
        ("pridict2", "library-diverse-invivo", "adv", "pe2"),
        ("pridict2", "trip-analysis", "k562", "pe2"),
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


def build_deeppe_scaffold_assignments() -> list[DatasheetScaffoldAssignment]:
    """
    Kim et al. 2021 used the sgRNA scaffold from pRG2 (Addgene 104174) for all libraries.

    That is the original Anzalone et al. 2019 pegRNA scaffold (catalog: conventional), not
    the Chen et al. 2021 optimized scaffold.
    """
    scaffold_id = SCAFFOLD_ID_CONVENTIONAL
    specs = [
        ("deeppe", "deeppe-ht", "hek293t", "pe2"),
        ("deeppe", "deeppe-type", "hek293t", "pe2"),
        ("deeppe", "deeppe-position", "hek293t", "pe2"),
        ("deeppe", "deeppe-endo", "hek293t", "pe2"),
        ("deeppe", "deeppe-endo", "hct116", "pe2"),
        ("deeppe", "deeppe-endo", "mda_mb_231", "pe2"),
    ]
    return [
        DatasheetScaffoldAssignment(
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            scaffold_id=scaffold_id,
            scaffold_source="deeppe_pgr2_scaffold",
        )
        for study, dataset, cell_line, pe_system in specs
    ]


def build_minsepie_scaffold_assignments(data_root: Optional[Path] = None) -> list[DatasheetScaffoldAssignment]:
    from ..utils.standardize_data import iter_minsepie_consolidated_datasheet_specs

    root = data_root or get_settings().data_root
    raw_path = root / "raw" / "minsepie" / "41587_2023_1678_MOESM5_ESM.tsv"
    if not raw_path.exists():
        return []

    rows: list[DatasheetScaffoldAssignment] = []
    for dataset, cell_line, pe_system in iter_minsepie_consolidated_datasheet_specs(root):
        scaffold_id = MINSEPIE_DATASET_SCAFFOLD_ID.get(dataset)
        if scaffold_id is None:
            logger.warning(
                "No scaffold mapping for minsepie dataset %s; skipping", dataset
            )
            continue
        rows.append(
            DatasheetScaffoldAssignment(
                study="minsepie",
                dataset=dataset,
                cell_line=cell_line,
                pe_system=pe_system,
                scaffold_id=scaffold_id,
                scaffold_source="minsepie_library_type",
            )
        )
    return rows


def build_optiprime_scaffold_assignments() -> list[DatasheetScaffoldAssignment]:
    """Hsu et al. 2026 used BlpI_F+E scaffold with tevoPreQ1 for all library screens."""
    scaffold_id = SCAFFOLD_ID_OPTIPRIME_BLPI_FE
    specs = [
        ("optiprime", "lib-mmr", "hek293t", "pe2"),
        ("optiprime", "lib-mmr", "hek293t", "pe4"),
        ("optiprime", "lib-mmr", "hela", "pe2"),
        ("optiprime", "lib-mmr", "hela", "pe4"),
        ("optiprime", "lib-mmr-controls", "hek293t", "pe2"),
        ("optiprime", "lib-mmr-controls", "hek293t", "pe4"),
        ("optiprime", "lib-mmr-controls", "hela", "pe2"),
        ("optiprime", "lib-mmr-controls", "hela", "pe4"),
        ("optiprime", "lib-cv", "hek293t", "pe2"),
        ("optiprime", "lib-cv", "hek293t", "pe4"),
        ("optiprime", "lib-cv", "hela", "pe2"),
        ("optiprime", "lib-cv", "hela", "pe4"),
    ]
    return [
        DatasheetScaffoldAssignment(
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            scaffold_id=scaffold_id,
            scaffold_source="optiprime_blpi_fe_scaffold",
        )
        for study, dataset, cell_line, pe_system in specs
    ]


def build_scaffold_assignments(data_root: Optional[Path] = None) -> list[DatasheetScaffoldAssignment]:
    """Infer scaffold_id for every known exported datasheet from raw study metadata."""
    return (
        build_deepprime_scaffold_assignments(data_root)
        + build_deeppe_scaffold_assignments()
        + build_pridict_scaffold_assignments()
        + build_minsepie_scaffold_assignments(data_root)
        + build_optiprime_scaffold_assignments()
    )


def _scaffold_assignment_lookup(
    assignments: list[DatasheetScaffoldAssignment],
) -> dict[tuple[str, str, str, str], int]:
    return {
        (
            _normalize_study_key(row.study),
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


def _parse_exported_datasheet_key(
    data_file: Path,
    exported_dir: Path,
) -> Optional[tuple[str, str, str, str]]:
    if not data_file.is_file() or data_file.suffix.lower() != ".csv":
        return None
    rel_parts = data_file.relative_to(exported_dir).parts
    if len(rel_parts) != 3 or "-" not in data_file.stem:
        return None
    study_key, dataset_name, _ = rel_parts
    cell_line, pe_system = data_file.stem.rsplit("-", 1)
    return (
        _normalize_study_key(study_key),
        _normalize_dataset_name(dataset_name),
        _normalize_datasheet_field(cell_line),
        _normalize_datasheet_field(pe_system),
    )


def _collect_exported_datasheet_keys(
    exported_dir: Path,
    dataset_index: dict[tuple[str, str], Dataset],
) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not exported_dir.exists():
        return keys
    for data_file in exported_dir.rglob("*.csv"):
        parsed = _parse_exported_datasheet_key(data_file, exported_dir)
        if parsed is None:
            continue
        study_key, dataset_name, cell_line, pe_system = parsed
        if (study_key, dataset_name) not in dataset_index:
            continue
        keys.add(parsed)
    return keys


def _prune_unregistered_datasets(
    session: Session,
    dataset_index: dict[tuple[str, str], Dataset],
) -> int:
    """Remove Dataset rows (and their datasheets) that are no longer in the registry."""
    registered_ids = {dataset.id for dataset in dataset_index.values()}
    removed = 0
    for dataset in session.scalars(select(Dataset)).all():
        if dataset.id in registered_ids:
            continue
        for datasheet in list(dataset.datasheets):
            session.delete(datasheet)
        session.delete(dataset)
        removed += 1
    return removed


def _prune_stale_datasheets(
    session: Session,
    valid_keys: set[tuple[str, str, str, str]],
) -> int:
    """Remove Datasheet rows that do not correspond to an exported CSV on disk."""
    removed = 0
    rows = session.scalars(
        select(Datasheet).options(
            joinedload(Datasheet.dataset).joinedload(Dataset.study)
        )
    ).unique().all()
    for row in rows:
        if row.dataset is None or row.dataset.study is None:
            session.delete(row)
            removed += 1
            continue
        key = (
            _normalize_study_key(row.dataset.study.name),
            _normalize_dataset_name(row.dataset.name),
            row.cell_line,
            row.pe_system,
        )
        if key not in valid_keys:
            session.delete(row)
            removed += 1
    return removed


def _index_exported_datasheets(
    session: Session,
    dataset_index: dict[tuple[str, str], Dataset],
    scaffold_by_datasheet: dict[tuple[str, str, str, str], int],
    *,
    data_root: Path,
    exported_dir: Path,
) -> int:
    """Register one Datasheet row per CSV under ``datasets/exported/``."""
    updated = 0
    if not exported_dir.exists():
        return updated

    for data_file in exported_dir.rglob("*.csv"):
        parsed = _parse_exported_datasheet_key(data_file, exported_dir)
        if parsed is None:
            continue
        study_key, dataset_name, cell_line, pe_system = parsed

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
            scaffold_id = SCAFFOLD_ID_OPTIMIZED

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
        pruned_datasets = _prune_unregistered_datasets(session, dataset_index)
        valid_keys = _collect_exported_datasheet_keys(exported_dir, dataset_index)
        count = _index_exported_datasheets(
            session,
            dataset_index,
            scaffold_by_datasheet,
            data_root=root,
            exported_dir=exported_dir,
        )
        pruned_datasheets = _prune_stale_datasheets(session, valid_keys)
    logger.info(
        "Indexed %s exported datasheets into catalog "
        "(removed %s stale datasets, %s stale datasheets)",
        count,
        pruned_datasets,
        pruned_datasheets,
    )
    return count

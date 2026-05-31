"""Seed hand-maintained catalog tables (Study, Dataset, Scaffold)."""

from __future__ import annotations

import logging

from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..db.models import Dataset, Scaffold, Study
from ..db.session import get_engine, get_session, init_db
from .scaffolds import PEGRNA_SCAFFOLDS, SCAFFOLD_ID_CONVENTIONAL, SCAFFOLD_ID_MINSEPIE_18NT, SCAFFOLD_ID_MINSEPIE_CODON_VARIANT, SCAFFOLD_ID_OPTIMIZED
from .studies import DATASET_REGISTRY, STUDY_REGISTRY

logger = logging.getLogger(__name__)

# Older catalog DBs used string primary keys for scaffolds; remap before delete.
_LEGACY_SCAFFOLD_IDS: dict[str, int] = {
    "conventional": SCAFFOLD_ID_CONVENTIONAL,
    "optimized": SCAFFOLD_ID_OPTIMIZED,
    "minsepie_18nt": SCAFFOLD_ID_MINSEPIE_18NT,
    "minsepie_codon_variant": SCAFFOLD_ID_MINSEPIE_CODON_VARIANT,
}


def _migrate_dataset_catalog_columns(engine: Engine) -> None:
    """Add dataset metadata columns when upgrading an existing SQLite catalog."""
    if not engine.dialect.name == "sqlite":
        return
    inspector = inspect(engine)
    if "dataset" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("dataset")}
    additions = [
        ("pegRNA_delivery_method", "VARCHAR(64)"),
        ("pe_delivery_method", "VARCHAR(64)"),
        ("edit_scope", "VARCHAR(64)"),
        ("experimental_method", "VARCHAR(64)"),
        ("target_context", "VARCHAR(64)"),
        ("standardizable", "BOOLEAN NOT NULL DEFAULT 1"),
    ]
    with engine.begin() as connection:
        for name, column_type in additions:
            if name not in existing:
                connection.execute(text(f"ALTER TABLE dataset ADD COLUMN {name} {column_type}"))


def _upsert_scaffolds(session: Session) -> None:
    for scaffold in PEGRNA_SCAFFOLDS:
        existing = session.get(Scaffold, scaffold.scaffold_id)
        if existing is None:
            session.add(
                Scaffold(
                    id=scaffold.scaffold_id,
                    name=scaffold.name,
                    sequence=scaffold.sequence,
                    description=scaffold.description or None,
                )
            )
        else:
            existing.name = scaffold.name
            existing.sequence = scaffold.sequence
            existing.description = scaffold.description or None


def _migrate_legacy_scaffolds(session: Session) -> None:
    """Remap legacy string scaffold IDs and drop duplicate scaffold rows."""
    for legacy_id, canonical_id in _LEGACY_SCAFFOLD_IDS.items():
        session.execute(
            text(
                "UPDATE datasheet SET scaffold_id = :canonical_id "
                "WHERE CAST(scaffold_id AS TEXT) = :legacy_id"
            ),
            {"canonical_id": canonical_id, "legacy_id": legacy_id},
        )

    canonical_ids = {scaffold.scaffold_id for scaffold in PEGRNA_SCAFFOLDS}
    removed = 0
    for row in list(session.scalars(select(Scaffold)).all()):
        try:
            row_id = int(row.id)
        except (TypeError, ValueError):
            session.delete(row)
            removed += 1
            continue
        if row_id not in canonical_ids:
            session.delete(row)
            removed += 1
    if removed:
        logger.info("Removed %s legacy scaffold row(s)", removed)


def _upsert_studies_and_datasets(session: Session) -> dict[tuple[str, str], Dataset]:
    study_ids: dict[str, int] = {}
    for record in STUDY_REGISTRY:
        study = session.scalar(select(Study).where(Study.name == record.key))
        if study is None:
            study = Study(
                name=record.key,
                publication_date=record.publication_date,
                authors=record.authors,
            )
            session.add(study)
            session.flush()
        else:
            study.publication_date = record.publication_date
            study.authors = record.authors
        study_ids[record.key] = study.id

    dataset_index: dict[tuple[str, str], Dataset] = {}
    for record in DATASET_REGISTRY:
        study_id = study_ids[record.study_key]
        dataset = session.scalar(
            select(Dataset).where(
                Dataset.study_id == study_id,
                Dataset.name == record.name,
            )
        )
        if dataset is None:
            dataset = Dataset(
                name=record.name,
                description=record.description,
                pegRNA_delivery_method=record.pegRNA_delivery_method,
                pe_delivery_method=record.pe_delivery_method,
                edit_scope=record.edit_scope,
                experimental_method=record.experimental_method,
                target_context=record.target_context,
                standardizable=record.standardizable,
                study_id=study_id,
            )
            session.add(dataset)
            session.flush()
        else:
            dataset.description = record.description
            dataset.pegRNA_delivery_method = record.pegRNA_delivery_method
            dataset.pe_delivery_method = record.pe_delivery_method
            dataset.edit_scope = record.edit_scope
            dataset.experimental_method = record.experimental_method
            dataset.target_context = record.target_context
            dataset.standardizable = record.standardizable
        dataset_index[(record.study_key, record.name)] = dataset
    return dataset_index


def init_catalog() -> None:
    """
    **Seed** the catalog database with hand-maintained reference data.

    "Seeding" means writing the static rows you define in code into empty (or
    existing) SQL tables — like planting fixed lookup tables before any dynamic
    data is added:

    - ``Study`` rows from ``catalog/studies.py`` (names, authors, publication dates)
    - ``Dataset`` rows from ``catalog/studies.py`` (pegRNA/PE delivery, edit scope, experimental method, target context)
    - ``Scaffold`` rows from ``catalog/scaffolds.py`` (pegRNA scaffold id, name, sequence)

    This step does **not** read CSV files, export raw data, standardize edits, or
    populate ``Datasheet`` rows. Those happen later in ``initialize_database()``.
    """
    init_db()
    _migrate_dataset_catalog_columns(get_engine())
    with get_session() as session:
        _upsert_scaffolds(session)
        _migrate_legacy_scaffolds(session)
        _upsert_studies_and_datasets(session)
    logger.info("Catalog seed complete (studies, datasets, scaffolds)")

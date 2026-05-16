"""Seed hand-maintained catalog tables (Study, Dataset, Scaffold)."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Dataset, Scaffold, Study
from ..db.session import get_session, init_db
from .scaffolds import PEGRNA_SCAFFOLDS
from .studies import DATASET_REGISTRY, STUDY_REGISTRY

logger = logging.getLogger(__name__)


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
                assay_type=record.assay_type,
                study_id=study_id,
            )
            session.add(dataset)
            session.flush()
        else:
            dataset.description = record.description
            dataset.assay_type = record.assay_type
        dataset_index[(record.study_key, record.name)] = dataset
    return dataset_index


def init_catalog() -> None:
    """
    **Seed** the catalog database with hand-maintained reference data.

    "Seeding" means writing the static rows you define in code into empty (or
    existing) SQL tables — like planting fixed lookup tables before any dynamic
    data is added:

    - ``Study`` rows from ``catalog/studies.py`` (names, authors, publication dates)
    - ``Dataset`` rows from ``catalog/studies.py`` (library names, assay types)
    - ``Scaffold`` rows from ``catalog/scaffolds.py`` (pegRNA scaffold id, name, sequence)

    This step does **not** read CSV files, export raw data, standardize edits, or
    populate ``Datasheet`` rows. Those happen later in ``initialize_database()``.
    """
    init_db()
    with get_session() as session:
        _upsert_scaffolds(session)
        _upsert_studies_and_datasets(session)
    logger.info("Catalog seed complete (studies, datasets, scaffolds)")

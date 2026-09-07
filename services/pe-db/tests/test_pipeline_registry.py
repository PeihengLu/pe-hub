"""Registry inventory: a new study/dataset must register a pipeline."""
from __future__ import annotations

from pe_db.catalog.studies import DATASET_REGISTRY, STUDY_REGISTRY
from pe_db.pipeline.names import normalize_name
from pe_db.pipeline.registry import get_standardizer, iter_pipelines
from pe_db.studies import load_studies


def test_every_catalog_study_has_a_pipeline():
    load_studies()
    catalog = {record.key for record in STUDY_REGISTRY}
    pipelines = {pipeline.key for pipeline in iter_pipelines()}
    assert catalog == pipelines


def test_every_standardizable_dataset_has_a_standardizer():
    load_studies()
    missing: list[str] = []
    for record in DATASET_REGISTRY:
        if not record.produces_standardized():
            continue
        try:
            get_standardizer(record.study_key, normalize_name(record.name))
        except ValueError:
            missing.append(f"{record.study_key}/{record.name}")
    assert missing == []

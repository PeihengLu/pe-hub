"""Static study and dataset metadata for the PE Database catalog.

Should be refactored to be more extensible if the study count grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class StudyRecord:
    key: str
    display_name: str
    publication_date: Optional[date]
    authors: str


@dataclass(frozen=True)
class DatasetRecord:
    study_key: str
    name: str
    description: str
    assay_type: str


STUDY_REGISTRY: tuple[StudyRecord, ...] = (
    StudyRecord(
        key="deepprime",
        display_name="DeepPrime",
        publication_date=date(2023, 5, 1),
        authors="Yu et al.",
    ),
    StudyRecord(
        key="pridict1",
        display_name="PRIDICT1",
        publication_date=date(2023, 8, 1),
        authors="Gerald et al.",
    ),
    StudyRecord(
        key="pridict2",
        display_name="PRIDICT2",
        publication_date=date(2024, 6, 1),
        authors="Mathis et al.",
    ),
    StudyRecord(
        key="minsepie",
        display_name="MinsePIE",
        publication_date=date(2023, 10, 1),
        authors="Koeppel et al.",
    ),
    StudyRecord(
        key="deeppe",
        display_name="DeepPE",
        publication_date=date(2021, 2, 1),
        authors="Kim et al.",
    ),
)

DATASET_REGISTRY: tuple[DatasetRecord, ...] = (
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-clinvar",
        description="DeepPrime ClinVar on-target library.",
        assay_type="lentiviral_ontarget",
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-small",
        description="DeepPrime small validation library (multi cell line / PE system).",
        assay_type="lentiviral_ontarget",
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-off",
        description="DeepPrime off-target library.",
        assay_type="offtarget",
    ),
    DatasetRecord(
        study_key="deepprime",
        name="deepprime-off-subpool",
        description="DeepPrime off-target sub-pools.",
        assay_type="offtarget",
    ),
    DatasetRecord(
        study_key="pridict1",
        name="library1",
        description="PRIDICT1 high-throughput screening library (~92k pegRNAs).",
        assay_type="lentiviral_ontarget",
    ),
    DatasetRecord(
        study_key="pridict2",
        name="library-diverse",
        description="PRIDICT2 diverse lentiviral library (in vitro cell lines).",
        assay_type="lentiviral_ontarget",
    ),
    DatasetRecord(
        study_key="pridict2",
        name="library-diverse-invivo",
        description="PRIDICT2 diverse library measured in vivo (AdV).",
        assay_type="in_vivo",
    ),
    DatasetRecord(
        study_key="minsepie",
        name="library-insert",
        description="MinSePIE insertion efficiency screens.",
        assay_type="insertion_library",
    ),
)

_STUDY_BY_KEY = {study.key: study for study in STUDY_REGISTRY}
_DATASET_BY_KEY = {(d.study_key, d.name): d for d in DATASET_REGISTRY}


def get_study_record(study_key: str) -> Optional[StudyRecord]:
    return _STUDY_BY_KEY.get(study_key.strip().lower())


def get_dataset_record(study_key: str, dataset_name: str) -> Optional[DatasetRecord]:
    return _DATASET_BY_KEY.get((study_key.strip().lower(), dataset_name.strip().lower()))

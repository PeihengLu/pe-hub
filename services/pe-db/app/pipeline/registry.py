"""Study pipeline registry: exporters, standardizers, scaffold assignments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..catalog.records import DatasheetScaffoldAssignment
from .names import canonical_dataset_name, normalize_name

Exporter = Callable[[], None]
Standardizer = Callable[[pd.DataFrame, str, str, str], None]
ScaffoldBuilder = Callable[[Optional[Path]], list[DatasheetScaffoldAssignment]]


@dataclass
class StudyPipeline:
    key: str
    exporters: tuple[Exporter, ...]
    standardizers: dict[str, Standardizer]
    scaffold_assignments: Optional[ScaffoldBuilder] = None


_PIPELINES: dict[str, StudyPipeline] = {}


def register_study(pipeline: StudyPipeline) -> None:
    key = pipeline.key.strip().lower()
    _PIPELINES[key] = pipeline


def get_pipeline(study: str) -> StudyPipeline:
    key = str(study).strip().lower()
    pipeline = _PIPELINES.get(key)
    if pipeline is None:
        raise ValueError(f"Study={study} not supported")
    return pipeline


def iter_pipelines() -> tuple[StudyPipeline, ...]:
    return tuple(_PIPELINES[key] for key in sorted(_PIPELINES))


def get_standardizer(study: str, dataset: str) -> Standardizer:
    pipeline = get_pipeline(study)
    normalized = normalize_name(dataset)
    hyphen = canonical_dataset_name(dataset)
    for name, fn in pipeline.standardizers.items():
        if normalize_name(name) == normalized or canonical_dataset_name(name) == hyphen:
            return fn
    raise ValueError(
        f"Unsupported dataset for study={pipeline.key}: {dataset}. "
        f"Supported: {sorted(pipeline.standardizers)}"
    )

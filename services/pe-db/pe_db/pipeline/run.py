"""Export and standardize orchestration (study dispatch via the pipeline registry)."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from pe_common.constants import DATA_ROOT

from ..catalog.studies import get_dataset_record
from .names import _normalize_name
from .registry import get_pipeline, get_standardizer, iter_pipelines
from .schema import _attach_endo_coordinate_columns, _drop_unmeasured_efficiency_rows, endo_standard_columns

logger = logging.getLogger(__name__)


def _ensure_studies_loaded() -> None:
    from .. import studies as _studies

    _studies.load_studies()


def is_standardizable(study: str, dataset: str) -> bool:
    record = get_dataset_record(study, dataset)
    return bool(record and record.standardizable)


def is_partially_standardizable(study: str, dataset: str) -> bool:
    record = get_dataset_record(study, dataset)
    return bool(record and record.partial)


def _missing_exported_datasets(study_name: str) -> list[str]:
    from ..catalog.studies import DATASET_REGISTRY

    study_dir = DATA_ROOT / "exported" / study_name
    if not study_dir.is_dir():
        return sorted(
            {d.name for d in DATASET_REGISTRY if d.study_key == study_name}
        ) or ["<all>"]

    missing: list[str] = []
    for record in DATASET_REGISTRY:
        if record.study_key != study_name:
            continue
        candidates = {record.name, _normalize_name(record.name)}
        if not any(
            (study_dir / candidate).is_dir() and any((study_dir / candidate).glob("*.csv"))
            for candidate in candidates
        ):
            missing.append(record.name)
    return sorted(missing)


def export_original_data(study: Optional[str] = None, force_reexport: bool = False) -> None:
    _ensure_studies_loaded()
    pipelines = {p.key: p for p in iter_pipelines()}
    target_studies = sorted(pipelines) if study is None else [study]
    if force_reexport:
        from ..formatted_cache import clear_formatted_cache

        clear_formatted_cache(study=study)
    for study_name in target_studies:
        pipeline = get_pipeline(study_name)
        missing = _missing_exported_datasets(study_name)
        if force_reexport or missing:
            if missing and not force_reexport:
                logger.info(
                    "Exporting study=%s (missing dataset(s): %s)",
                    study_name,
                    ", ".join(missing),
                )
            else:
                logger.info("Exporting study=%s", study_name)
            for exporter in pipeline.exporters:
                exporter()
        else:
            logger.info("Study=%s already exported", study_name)

    from ..catalog.datasheets import index_exported_datasheets

    index_exported_datasheets()


def standardize_exported_data(
    study: Optional[str] = None,
    *,
    force: bool = False,
) -> int:
    _ensure_studies_loaded()
    exported_root = DATA_ROOT / "exported"
    if not exported_root.exists():
        logger.warning("No exported data directory at %s", exported_root)
        return 0

    studies = (
        sorted(p.key for p in iter_pipelines())
        if study is None
        else [study.strip().lower()]
    )
    if force:
        from ..formatted_cache import clear_formatted_cache

        clear_formatted_cache(study=study)
    count = 0
    failures: list[str] = []
    for study_key in studies:
        study_dir = exported_root / study_key
        if not study_dir.is_dir():
            continue
        for csv_path in study_dir.rglob("*.csv"):
            if not csv_path.is_file():
                continue
            rel_parts = csv_path.relative_to(study_dir).parts
            if len(rel_parts) != 2 or "-" not in csv_path.stem:
                continue
            dataset_name, _filename = rel_parts
            cell_line, pe_system = csv_path.stem.rsplit("-", 1)
            normalized_dataset = _normalize_name(dataset_name)
            output_path = (
                DATA_ROOT / "standardized" / study_key / normalized_dataset
                / f"{_normalize_name(cell_line)}-{_normalize_name(pe_system)}.parquet"
            )
            if not (
                is_standardizable(study_key, dataset_name)
                or is_partially_standardizable(study_key, dataset_name)
            ):
                logger.debug(
                    "Skipping standardization for %s/%s (export-only or unsupported schema)",
                    study_key,
                    dataset_name,
                )
                continue
            if output_path.exists() and not force:
                logger.debug("Already standardized: %s", output_path)
                continue
            logger.info(
                "Standardizing %s/%s %s-%s",
                study_key,
                dataset_name,
                cell_line,
                pe_system,
            )
            try:
                standardize_pe_data(
                    study=study_key,
                    dataset=dataset_name,
                    cell_line=cell_line,
                    pe_system=pe_system,
                )
            except Exception as exc:
                logger.error(
                    "Failed to standardize %s/%s %s-%s: %s",
                    study_key,
                    dataset_name,
                    cell_line,
                    pe_system,
                    exc,
                )
                failures.append(f"{study_key}/{dataset_name} {cell_line}-{pe_system}: {exc}")
                continue
            count += 1
    if failures:
        logger.error(
            "Standardized %s datasheet(s); %s failed:\n  %s",
            count,
            len(failures),
            "\n  ".join(failures),
        )
    else:
        logger.info("Standardized %s exported datasheet(s)", count)
    return count


def standardize_pe_data(
    *,
    study: str,
    cell_line: str,
    pe_system: str,
    dataset: str,
) -> pd.DataFrame:
    _ensure_studies_loaded()
    study = _normalize_name(study)
    cell_line = _normalize_name(cell_line)
    pe_system = _normalize_name(pe_system)
    dataset = str(dataset).strip().lower()
    normalized_dataset = _normalize_name(dataset)
    if not (is_standardizable(study, dataset) or is_partially_standardizable(study, dataset)):
        raise ValueError(
            f"Dataset {study}/{dataset} is not standardizable "
            "(full or partial entry-level path unavailable)."
        )
    dataset_candidates = [dataset]
    if normalized_dataset not in dataset_candidates:
        dataset_candidates.append(normalized_dataset)
    hyphenated = normalized_dataset.replace("_", "-")
    if hyphenated not in dataset_candidates:
        dataset_candidates.append(hyphenated)

    input_path = None
    for dataset_candidate in dataset_candidates:
        candidate_path = (
            DATA_ROOT / "exported" / study / dataset_candidate / f"{cell_line}-{pe_system}.csv"
        )
        if candidate_path.exists():
            input_path = candidate_path
            break
    if input_path is None:
        raise FileNotFoundError(
            "Exported input file not found for any dataset variant: "
            f"{dataset_candidates} (study={study}, cell_line={cell_line}, pe_system={pe_system})"
        )

    data = pd.read_csv(input_path, low_memory=False)
    get_standardizer(study, normalized_dataset)(data, cell_line, pe_system, normalized_dataset)

    output_path = (
        DATA_ROOT / "standardized" / study / normalized_dataset
        / f"{cell_line}-{pe_system}.parquet"
    )
    if not output_path.exists():
        raise FileNotFoundError(
            "Standardization did not produce expected output file: "
            f"{output_path}"
        )
    result = pd.read_parquet(output_path)
    needs_rewrite = False
    record = get_dataset_record(study, dataset)
    if record is not None and record.target_context == "endogenous":
        missing = any(column not in result.columns for column in endo_standard_columns)
        obsolete = any(
            column.startswith("endo_") and column not in endo_standard_columns
            for column in result.columns
        )
        if missing or obsolete:
            result = _attach_endo_coordinate_columns(result)
            needs_rewrite = True

    result, n_dropped = _drop_unmeasured_efficiency_rows(
        result, label=f"{study}/{normalized_dataset} {cell_line}-{pe_system}"
    )
    if n_dropped:
        needs_rewrite = True

    if needs_rewrite:
        result.to_parquet(output_path, index=False)
    return result

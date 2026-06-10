"""Disk cache for standardized -> model-format conversions."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from pe_common.constants import DATA_ROOT

logger = logging.getLogger(__name__)

FORMATTED_MODEL_FORMATS = frozenset({"deepprime", "pridict", "pridict2", "oped"})


def _normalize_segment(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def formatted_root(datasets_dir: Optional[Path] = None) -> Path:
    return (datasets_dir or DATA_ROOT) / "formatted"


def formatted_cache_path(
    target_format: str,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Path:
    if target_format not in FORMATTED_MODEL_FORMATS:
        raise ValueError(f"Unsupported formatted cache format: {target_format}")
    study_key = _normalize_segment(study)
    dataset_key = _normalize_segment(dataset)
    cell_line_key = _normalize_segment(cell_line)
    pe_system_key = _normalize_segment(pe_system)
    return (
        formatted_root(datasets_dir)
        / target_format
        / study_key
        / dataset_key
        / f"{cell_line_key}-{pe_system_key}.parquet"
    )


def clear_formatted_cache(
    study: Optional[str] = None,
    *,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Remove cached model-format parquet files.

    Called when ``force_reexport`` or ``force_standardize`` rebuilds upstream data.
    """
    root = formatted_root(datasets_dir)
    if not root.is_dir():
        return
    if study is None:
        shutil.rmtree(root)
        logger.info("Cleared formatted model cache")
        return

    study_key = _normalize_segment(study)
    for format_dir in root.iterdir():
        if not format_dir.is_dir():
            continue
        target = format_dir / study_key
        if target.is_dir():
            shutil.rmtree(target)
    logger.info("Cleared formatted model cache for study=%s", study)


def load_formatted_cache(
    target_format: str,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
    *,
    datasets_dir: Optional[Path] = None,
    expected_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    path = formatted_cache_path(
        target_format,
        study,
        dataset,
        cell_line,
        pe_system,
        datasets_dir=datasets_dir,
    )
    if not path.is_file():
        return None
    cached = pd.read_parquet(path)
    if expected_rows is not None and len(cached) != expected_rows:
        logger.warning(
            "Formatted cache row count mismatch for %s (%s != %s); reconverting",
            path,
            len(cached),
            expected_rows,
        )
        return None
    logger.info("Loaded formatted cache: %s", path)
    return cached


def save_formatted_cache(
    df: pd.DataFrame,
    target_format: str,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Path:
    path = formatted_cache_path(
        target_format,
        study,
        dataset,
        cell_line,
        pe_system,
        datasets_dir=datasets_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote formatted cache: %s", path)
    return path

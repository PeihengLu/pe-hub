"""Disk cache for standardized -> model-format conversions."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from pe_common.constants import DATA_ROOT

from .format_registry import known_model_formats

logger = logging.getLogger(__name__)

# Bump a format when its converter semantics change so stale parquet is skipped.
# Formats still at revision 1 keep pre-revision caches (no ``.revision`` file).
FORMATTED_CACHE_REVISIONS: dict[str, int] = {
    "deepprime": 1,
    "pridict": 1,
    "pridict2": 1,
    "oped": 2,  # PBS from WT, RT from Mut
    "optiprime": 1,
}


def formatted_model_formats() -> frozenset[str]:
    """Model format names eligible for disk cache (derived from the format registry)."""
    return known_model_formats()


def _normalize_segment(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def formatted_root(datasets_dir: Optional[Path] = None) -> Path:
    return (datasets_dir or DATA_ROOT) / "formatted"


def formatted_cache_revision(target_format: str) -> int:
    return FORMATTED_CACHE_REVISIONS.get(target_format, 1)


def _entry_revision_path(parquet_path: Path) -> Path:
    """Per-file sidecar. A format-level ``.revision`` is not used: saving one
    sheet must not mark sibling parquets as current."""
    return parquet_path.with_name(parquet_path.name + ".revision")


def _read_entry_revision(parquet_path: Path) -> Optional[int]:
    path = _entry_revision_path(parquet_path)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (TypeError, ValueError):
        return None


def _write_entry_revision(parquet_path: Path, target_format: str) -> None:
    path = _entry_revision_path(parquet_path)
    path.write_text(str(formatted_cache_revision(target_format)), encoding="utf-8")


def _entry_revision_is_current(parquet_path: Path, target_format: str) -> bool:
    expected = formatted_cache_revision(target_format)
    stored = _read_entry_revision(parquet_path)
    if stored is None:
        # Pre-revision caches are valid only while the format is still at 1.
        return expected == 1
    return stored == expected


def formatted_cache_path(
    target_format: str,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
    *,
    datasets_dir: Optional[Path] = None,
) -> Path:
    if target_format not in formatted_model_formats():
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
    if not _entry_revision_is_current(path, target_format):
        logger.warning(
            "Formatted cache revision mismatch for %s (have %s, want %s); reconverting",
            path,
            _read_entry_revision(path),
            formatted_cache_revision(target_format),
        )
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
    _write_entry_revision(path, target_format)
    logger.info("Wrote formatted cache: %s", path)
    return path

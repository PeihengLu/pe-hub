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
    "deepprime": 2,  # WT74 from unpadded WT; PBS/RTT drop alignment pads
    "pridict": 2,  # drop N pads; map coords back to the author frame
    "pridict2": 2,
    "oped": 3,  # 47 bp crop from grown aligned sequences
    "optiprime": 2,  # spacer at 0:20; drop pads in PBS/RTT / full sequences
}


def formatted_model_formats() -> frozenset[str]:
    """Model format names eligible for disk cache (derived from the format registry)."""
    return known_model_formats()


def _normalize_segment(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def formatted_root(datasets_dir: Optional[Path] = None) -> Path:
    return (datasets_dir or DATA_ROOT) / "formatted"


def standardized_root(datasets_dir: Optional[Path] = None) -> Path:
    return (datasets_dir or DATA_ROOT) / "standardized"


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


def _tree_stats(path: Path) -> tuple[int, int]:
    files = 0
    nbytes = 0
    if path.is_file():
        return 1, path.stat().st_size
    if not path.is_dir():
        return 0, 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            nbytes += child.stat().st_size
    return files, nbytes


def _formatted_clear_targets(
    *,
    study: Optional[str],
    target_format: Optional[str],
    datasets_dir: Optional[Path],
) -> list[Path]:
    root = formatted_root(datasets_dir)
    if not root.is_dir():
        return []
    if target_format is not None:
        format_key = _normalize_segment(target_format)
        if format_key not in formatted_model_formats():
            raise ValueError(f"Unsupported formatted cache format: {target_format}")
        format_dirs = [root / format_key]
    else:
        format_dirs = [path for path in root.iterdir() if path.is_dir()]
    if study is None and target_format is None:
        return [root]
    if study is None:
        return [path for path in format_dirs if path.exists()]
    study_key = _normalize_segment(study)
    return [path / study_key for path in format_dirs if (path / study_key).exists()]


def _standardized_clear_targets(
    *,
    study: Optional[str],
    datasets_dir: Optional[Path],
) -> list[Path]:
    root = standardized_root(datasets_dir)
    if not root.is_dir():
        return []
    if study is None:
        return [root]
    target = root / _normalize_segment(study)
    return [target] if target.exists() else []


def describe_cache_targets(paths: list[Path]) -> list[dict[str, int | str]]:
    """Return ``path`` / file count / byte size for each cache root."""
    rows: list[dict[str, int | str]] = []
    for path in paths:
        files, nbytes = _tree_stats(path)
        rows.append({"path": str(path), "files": files, "bytes": nbytes})
    return rows


def clear_cached_data(
    *,
    formatted: bool = True,
    standardized: bool = False,
    study: Optional[str] = None,
    target_format: Optional[str] = None,
    datasets_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Remove derived PE-DB caches (model-format parquet and/or standardized parquet).

    Does not touch ``raw/``, ``exported/``, ``catalog/``, or reference genomes.
    """
    if target_format is not None and not formatted:
        raise ValueError("--format applies to the formatted cache; pass formatted=True")
    targets: list[Path] = []
    if formatted:
        targets.extend(
            _formatted_clear_targets(
                study=study, target_format=target_format, datasets_dir=datasets_dir
            )
        )
    if standardized:
        targets.extend(_standardized_clear_targets(study=study, datasets_dir=datasets_dir))

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_targets.append(path)

    report = {
        "dry_run": dry_run,
        "study": None if study is None else _normalize_segment(study),
        "format": None if target_format is None else _normalize_segment(target_format),
        "targets": describe_cache_targets(unique_targets),
        "removed": [] if dry_run else [str(path) for path in unique_targets],
    }
    if dry_run:
        logger.info("Dry-run: would clear %s cache root(s)", len(unique_targets))
        return report
    for path in unique_targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        logger.info("Cleared cache root: %s", path)
    return report


def clear_formatted_cache(
    study: Optional[str] = None,
    *,
    target_format: Optional[str] = None,
    datasets_dir: Optional[Path] = None,
) -> None:
    """Remove cached model-format parquet files.

    Called when ``force_reexport`` or ``force_standardize`` rebuilds upstream data.
    """
    report = clear_cached_data(
        formatted=True,
        standardized=False,
        study=study,
        target_format=target_format,
        datasets_dir=datasets_dir,
        dry_run=False,
    )
    n_roots = len(report["targets"])
    if study is None and target_format is None:
        logger.info("Cleared formatted model cache (%s root(s))", n_roots)
    elif target_format is None:
        logger.info("Cleared formatted model cache for study=%s", study)
    else:
        logger.info(
            "Cleared formatted model cache for study=%s format=%s",
            study,
            target_format,
        )


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

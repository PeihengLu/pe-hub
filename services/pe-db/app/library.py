"""Headless PE Database API (shared by HTTP handlers, the pe-db CLI, and pe-ensemble CLI)."""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

from pe_common.splits import SplitConfig, split_config_from_params

from .catalog.initialize import initialize_database
from .catalog.seed import init_catalog
from .converter import DataConverter
from .db.repository import CatalogRepository
from .db.session import get_session
from .format_registry import known_output_formats, validate_output_format
from .plugin_loader import load_active_plugins, reload_active_plugins
from .utils.standardize_data import export_original_data, standardize_exported_data

logger = logging.getLogger(__name__)

SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]
ProgressCallback = Callable[[str], None]

_plugins_loaded = False


class PeDbLibraryError(ValueError):
    """Raised when filter/export or catalog arguments are invalid."""


def ensure_plugins_loaded() -> list[str]:
    """Load active plugin converters once per process."""
    global _plugins_loaded
    if _plugins_loaded:
        from .plugin_loader import loaded_plugin_names

        return list(loaded_plugin_names())
    loaded = load_active_plugins()
    _plugins_loaded = True
    return loaded


def run_seed() -> None:
    """Create catalog tables and seed Study, Dataset, and Scaffold rows."""
    init_catalog()


def run_init(*, force_export: bool = False, force_standardize: bool = False) -> None:
    """Seed catalog, export raw study files, and standardize to parquet."""
    initialize_database(
        force_export=force_export,
        force_standardize=force_standardize,
    )
    ensure_plugins_loaded()


def run_export(
    *,
    study: Optional[str] = None,
    force_reexport: bool = False,
    standardize: bool = True,
    force_standardize: bool = False,
) -> dict[str, Any]:
    """Export raw study files and optionally standardize."""
    if study is None and not force_reexport and not force_standardize and standardize:
        run_init(force_export=force_reexport, force_standardize=force_standardize)
    else:
        export_original_data(study=study, force_reexport=force_reexport)
        if standardize:
            standardize_exported_data(study=study, force=force_standardize)
        ensure_plugins_loaded()
    with get_session() as session:
        count = len(CatalogRepository(session).list_datasheets())
    return {
        "status": "success",
        "study": study or "all",
        "force_reexport": force_reexport,
        "standardized": standardize,
        "datasheets_in_catalog": count,
    }


def run_standardize(*, study: Optional[str] = None, force: bool = False) -> dict[str, Any]:
    """Standardize exported CSVs to parquet."""
    standardize_exported_data(study=study, force=force)
    ensure_plugins_loaded()
    return {
        "status": "success",
        "study": study or "all",
        "force": force,
    }


def run_clear_cached_data(
    *,
    formatted: bool = True,
    standardized: bool = False,
    study: Optional[str] = None,
    target_format: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove derived formatted and/or standardized caches."""
    from .formatted_cache import clear_cached_data

    try:
        return clear_cached_data(
            formatted=formatted,
            standardized=standardized,
            study=study,
            target_format=target_format,
            dry_run=dry_run,
        )
    except ValueError as exc:
        raise PeDbLibraryError(str(exc)) from exc


def run_convert_sheet(
    *,
    study: str,
    dataset: str,
    cell_line: str,
    pe_system: str,
) -> dict[str, Any]:
    """Standardize one exported datasheet."""
    converter = DataConverter()
    result = converter.convert_to_standardized(
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
    )
    return {
        "status": "success",
        "message": f"Successfully standardized {study}/{dataset} data",
        "records_converted": len(result),
        "output_columns": list(result.columns),
    }


def reload_plugins() -> list[str]:
    """Reload active plugin converters from ``PLUGINS_ROOT``."""
    global _plugins_loaded
    loaded = reload_active_plugins()
    _plugins_loaded = True
    return loaded


def filter_data(
    *,
    study: Optional[list[str]] = None,
    dataset: Optional[list[str]] = None,
    cell_line: Optional[list[str]] = None,
    pe_system: Optional[list[str]] = None,
    edit_type: Optional[list[str]] = None,
    edit_length: Optional[list[int]] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[list[str]] = None,
    experimental_method: Optional[list[str]] = None,
    target_context: Optional[list[str]] = None,
    scaffold_name: Optional[list[str]] = None,
    format_: Optional[str] = None,
    split_strategy: Optional[SplitStrategy] = None,
    train_pct: Optional[float] = None,
    val_pct: Optional[float] = None,
    test_pct: Optional[float] = None,
    cv_folds: Optional[int] = None,
    use_original_fold: bool = False,
    original_fold_test_value: float = -1.0,
    split_random_state: int = 42,
    merge: bool = False,
    summary_only: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Filter catalog and/or export model-format data (same contract as ``GET /api/filter``)."""
    ensure_plugins_loaded()

    if summary_only and format_ is None:
        format_ = "std"

    if format_ is not None:
        format_ = validate_output_format(format_)

    if format_ is not None and split_strategy is None:
        raise PeDbLibraryError(
            "split_strategy is required when format is set (use 'none' for no split columns)."
        )

    split_config: Optional[SplitConfig] = None
    if format_ is not None:
        split_config = split_config_from_params(
            strategy=split_strategy,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            cv_folds=cv_folds,
            use_original_fold=use_original_fold,
            original_fold_test_value=original_fold_test_value,
            random_state=split_random_state,
        )

    with get_session() as session:
        result = CatalogRepository(session).filter_all(
            study_name=study,
            dataset_name=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
            edit_scope=edit_scope,
            experimental_method=experimental_method,
            target_context=target_context,
            scaffold_name=scaffold_name,
            target_format=format_,
            split_config=split_config,
            merge_groups=merge,
            summary_only=summary_only,
            progress_callback=progress_callback,
        )

    if format_ is None:
        return {
            "status": "success",
            "format": None,
            "count": len(result),
            "datasheets": result,
        }
    return {"status": "success", **result}


def _coerce_list_param(value: Any) -> Optional[list[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    return [value]


def filter_from_params(
    params: dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Run ``filter_data`` from a PE-DB query-param dict (HTTP or ensemble ``build_pe_db_filter_params``)."""
    token = params.get("progress_token")
    callback = progress_callback
    if callback is None and token:
        from pe_common.conversion_progress import append_progress, clear_progress

        clear_progress(str(token))

        def callback(message: str, _token: str = str(token)) -> None:
            append_progress(_token, message)

    return filter_data(
        study=_coerce_list_param(params.get("study")),
        dataset=_coerce_list_param(params.get("dataset")),
        cell_line=_coerce_list_param(params.get("cell_line")),
        pe_system=_coerce_list_param(params.get("pe_system")),
        edit_type=_coerce_list_param(params.get("edit_type")),
        edit_length=_coerce_list_param(params.get("edit_length")),
        edit_efficiency_min=params.get("edit_efficiency_min"),
        edit_efficiency_max=params.get("edit_efficiency_max"),
        edit_scope=_coerce_list_param(params.get("edit_scope")),
        experimental_method=_coerce_list_param(params.get("experimental_method")),
        target_context=_coerce_list_param(params.get("target_context")),
        scaffold_name=_coerce_list_param(params.get("scaffold_name")),
        format_=params.get("format"),
        split_strategy=params.get("split_strategy"),
        train_pct=params.get("train_pct"),
        val_pct=params.get("val_pct"),
        test_pct=params.get("test_pct"),
        cv_folds=params.get("cv_folds"),
        use_original_fold=bool(params.get("use_original_fold", False)),
        original_fold_test_value=float(params.get("original_fold_test_value", -1.0)),
        split_random_state=int(params.get("split_random_state", 42)),
        merge=bool(params.get("merge", False)),
        summary_only=bool(params.get("summary_only", False)),
        progress_callback=callback,
    )


def list_output_formats() -> tuple[str, ...]:
    """Return supported ``format`` values for filter/export."""
    return tuple(sorted(known_output_formats()))


def list_studies() -> list[dict[str, Any]]:
    """List catalog studies (same contract as ``GET /api/studies``)."""
    with get_session() as session:
        return [row.model_dump() for row in CatalogRepository(session).list_studies()]


def list_datasets(*, study: Optional[str] = None) -> list[dict[str, Any]]:
    """List catalog datasets (same contract as ``GET /api/datasets``)."""
    with get_session() as session:
        return [
            row.model_dump()
            for row in CatalogRepository(session).list_datasets(study_name=study)
        ]


def list_datasheets(
    *,
    study: Optional[str] = None,
    dataset: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List catalog datasheets (same contract as ``GET /api/datasheets``)."""
    with get_session() as session:
        return [
            row.model_dump()
            for row in CatalogRepository(session).list_datasheets(
                study_name=study,
                dataset_name=dataset,
            )
        ]


def list_scaffolds() -> list[dict[str, Any]]:
    """List pegRNA scaffolds (same contract as ``GET /api/scaffolds``)."""
    with get_session() as session:
        return [row.model_dump() for row in CatalogRepository(session).list_scaffolds()]


def catalog_statistics(
    *,
    edit_type: Optional[str] = None,
    edit_length: Optional[int] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[str] = None,
    experimental_method: Optional[str] = None,
    target_context: Optional[str] = None,
    scaffold_name: Optional[str] = None,
) -> dict[str, Any]:
    """Descriptive statistics over edit rows (same contract as ``GET /api/statistics``)."""
    with get_session() as session:
        stats = CatalogRepository(session).compute_statistics(
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
            edit_scope=edit_scope,
            experimental_method=experimental_method,
            target_context=target_context,
            scaffold_name=scaffold_name,
        )
    return stats.model_dump()

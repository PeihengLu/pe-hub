"""PE Database API - FastAPI application.

Catalog and filter routes used by PE Hub and PE-Ensemble. Admin pipeline
operations (init, export, convert) live on the ``pedb`` CLI / library.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db.schemas import DatasetRead, DatasheetRead, ScaffoldRead, StatisticsRead, StudyRead
from .library import (
    PeDbLibraryError,
    catalog_statistics,
    filter_from_params,
    list_datasheets as library_list_datasheets,
    list_datasets as library_list_datasets,
    list_scaffolds as library_list_scaffolds,
    list_studies as library_list_studies,
    reload_plugins as library_reload_plugins,
    run_init,
)

SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pe-db-sync")
    loop.set_default_executor(executor)
    try:
        await asyncio.to_thread(
            run_init,
            force_export=_env_flag("PE_DB_FORCE_EXPORT"),
            force_standardize=_env_flag("PE_DB_FORCE_STANDARDIZE"),
        )
        from .plugin_loader import loaded_plugin_names

        names = loaded_plugin_names()
        if names:
            logger.info("Loaded PE-DB plugins: %s", ", ".join(names))
        yield
    finally:
        from .process_pool import shutdown_mfe_process_pool

        shutdown_mfe_process_pool()
        executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(
    title="PE Database API",
    description="API for serving prime editing efficiency data and catalog metadata",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRACEFUL_SHUTDOWN_SECONDS = 5


@app.get("/")
async def root():
    settings = get_settings()
    return {
        "name": "PE Database API",
        "version": "0.2.0",
        "description": "Prime editing data service with relational catalog",
        "catalog_database": str(settings.catalog_db_path),
        "endpoints": {
            "studies": "/api/studies",
            "datasets": "/api/datasets",
            "datasheets": "/api/datasheets",
            "scaffolds": "/api/scaffolds",
            "filter": "/api/filter",
            "statistics": "/api/statistics",
            "health": "/health",
        },
    }


@app.get("/api/studies", response_model=list[StudyRead])
async def list_studies():
    """List all studies in the catalog."""
    return library_list_studies()


@app.get("/api/scaffolds", response_model=list[ScaffoldRead])
async def list_scaffolds():
    """List pegRNA scaffold definitions (id, name, sequence)."""
    return library_list_scaffolds()


@app.get("/api/datasets", response_model=list[DatasetRead])
async def list_catalog_datasets(
    study: Optional[str] = Query(None, description="Filter by study key (e.g. deepprime)"),
):
    """List datasets registered in the catalog."""
    return library_list_datasets(study=study)


@app.get("/api/datasheets", response_model=list[DatasheetRead])
async def list_datasheets(
    study: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
):
    """List datasheets (cell line × PE system) with scaffold and file metadata."""
    return library_list_datasheets(study=study, dataset=dataset)


@app.get("/api/filter")
async def filter_data(
    study: Optional[list[str]] = Query(None, description="Filter by study key (e.g. deepprime)."),
    dataset: Optional[list[str]] = Query(None, description="Filter by dataset name within the study."),
    cell_line: Optional[list[str]] = Query(None, description="Filter by cell line (e.g. HEK293T)."),
    pe_system: Optional[list[str]] = Query(None, description="Filter by PE system (e.g. PE2max)."),
    edit_type: Optional[list[str]] = Query(None, description="Filter edits by type (sub, ins, del)."),
    edit_length: Optional[list[int]] = Query(None, description="Filter edits by length."),
    edit_efficiency_min: Optional[float] = Query(None, description="Minimum editing efficiency."),
    edit_efficiency_max: Optional[float] = Query(None, description="Maximum editing efficiency."),
    edit_scope: Optional[list[str]] = Query(None, description="Filter by edit scope (on_target, off_target)."),
    experimental_method: Optional[list[str]] = Query(None, description="Filter by experimental method."),
    target_context: Optional[list[str]] = Query(None, description="Filter by target context."),
    scaffold_name: Optional[list[str]] = Query(None, description="Filter by pegRNA scaffold name."),
    format_: Optional[str] = Query(
        None,
        alias="format",
        description=(
            "Output format. When unset, returns matching datasheets as usual. "
            "When set, returns standardizable datasets' data converted from the "
            "standardized schema into the requested model format."
        ),
    ),
    split_strategy: Optional[SplitStrategy] = Query(
        None,
        description=(
            "Required when format is set. Split assignment strategy: "
            "none, holdout_2, holdout_3, or cv."
        ),
    ),
    train_pct: Optional[float] = Query(None),
    val_pct: Optional[float] = Query(None),
    test_pct: Optional[float] = Query(None),
    cv_folds: Optional[int] = Query(None),
    use_original_fold: bool = Query(
        False,
        description="When true, use author original_fold assignments where available.",
    ),
    original_fold_test_value: float = Query(
        -1.0,
        description=(
            "original_fold value treated as the test partition when use_original_fold is true "
            "(-1 for DeepPrime-style held-out test; 0–4 for PRIDICT2 CV test folds)."
        ),
    ),
    split_random_state: int = Query(42),
    merge: bool = Query(
        False,
        description=(
            "When true, merge all matching datasheets before split assignment. "
            "Reassigns group_id by shared protospacer after merge."
        ),
    ),
    summary_only: bool = Query(
        False,
        description=(
            "When true, return standardized-data counts and split summaries only. "
            "Skips model-format conversion and omits per-row records for faster preview."
        ),
    ),
    progress_token: Optional[str] = Query(
        None,
        min_length=8,
        max_length=64,
        description=(
            "Optional progress token. When set, coarse conversion progress lines are "
            "appended to a shared progress log file for the caller to tail."
        ),
    ),
):
    """Filter datasheets by catalog/edit metadata; optionally emit model-format data."""
    params = {
        "study": study,
        "dataset": dataset,
        "cell_line": cell_line,
        "pe_system": pe_system,
        "edit_type": edit_type,
        "edit_length": edit_length,
        "edit_efficiency_min": edit_efficiency_min,
        "edit_efficiency_max": edit_efficiency_max,
        "edit_scope": edit_scope,
        "experimental_method": experimental_method,
        "target_context": target_context,
        "scaffold_name": scaffold_name,
        "format": format_,
        "split_strategy": split_strategy,
        "train_pct": train_pct,
        "val_pct": val_pct,
        "test_pct": test_pct,
        "cv_folds": cv_folds,
        "use_original_fold": use_original_fold,
        "original_fold_test_value": original_fold_test_value,
        "split_random_state": split_random_state,
        "merge": merge,
        "summary_only": summary_only,
        "progress_token": progress_token,
    }
    try:
        return await asyncio.to_thread(filter_from_params, params)
    except PeDbLibraryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error filtering data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error filtering data: {exc}") from exc


@app.get("/api/statistics", response_model=StatisticsRead)
async def get_statistics(
    edit_type: Optional[str] = Query(
        None, description="Filter entries by edit type (sub, ins, del)."
    ),
    edit_length: Optional[int] = Query(None, description="Filter entries by edit length."),
    edit_efficiency_min: Optional[float] = Query(
        None, description="Minimum editing efficiency (inclusive)."
    ),
    edit_efficiency_max: Optional[float] = Query(
        None, description="Maximum editing efficiency (inclusive)."
    ),
    edit_scope: Optional[str] = Query(
        None, description="Filter by dataset edit scope (on_target, off_target)."
    ),
    experimental_method: Optional[str] = Query(
        None, description="Filter by experimental method (in_vitro, in_vivo)."
    ),
    target_context: Optional[str] = Query(
        None, description="Filter by target context (endogenous, non_endogenous)."
    ),
    scaffold_name: Optional[str] = Query(None, description="Filter by pegRNA scaffold name."),
):
    """Descriptive statistics over edit rows, with optional catalog and entry filters."""
    try:
        return catalog_statistics(
            edit_type=edit_type,
            edit_length=edit_length,
            edit_efficiency_min=edit_efficiency_min,
            edit_efficiency_max=edit_efficiency_max,
            edit_scope=edit_scope,
            experimental_method=experimental_method,
            target_context=target_context,
            scaffold_name=scaffold_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error computing statistics: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Error computing statistics: {exc}"
        ) from exc


@app.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "catalog_database": str(settings.catalog_db_path),
        "catalog_database_exists": settings.catalog_db_path.exists(),
    }


@app.post("/api/plugins/reload")
async def reload_plugin_formats():
    """Reload active plugin converters from ``PLUGINS_ROOT``."""
    loaded = await asyncio.to_thread(library_reload_plugins)
    return {"loaded": loaded, "count": len(loaded)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )

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
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pe_common.filter_params import (
    CatalogFilterQuery,
    SplitExportQuery,
    query_dependency,
)

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_catalog_filters = query_dependency(CatalogFilterQuery)
_split_export = query_dependency(SplitExportQuery)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pe-db-sync")
    loop.set_default_executor(executor)
    try:
        force_export = _env_flag("PE_DB_FORCE_EXPORT")
        force_standardize = _env_flag("PE_DB_FORCE_STANDARDIZE")
        logger.info(
            "Startup init force_export=%s force_standardize=%s",
            force_export,
            force_standardize,
        )
        await asyncio.to_thread(
            run_init,
            force_export=force_export,
            force_standardize=force_standardize,
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
    filters: Annotated[CatalogFilterQuery, Depends(_catalog_filters)],
    split: Annotated[SplitExportQuery, Depends(_split_export)],
    format_: Optional[str] = Query(
        None,
        alias="format",
        description=(
            "Output format. When unset, returns matching datasheets as usual. "
            "When set, returns standardizable datasets' data converted from the "
            "standardized schema into the requested model format."
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
        **filters.model_dump(),
        **split.model_dump(),
        "format": format_,
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

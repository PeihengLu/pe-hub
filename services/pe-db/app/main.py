"""PE Database API - Main FastAPI Application

Serves prime editing efficiency data and exposes the catalog schema defined in
``diagrams/illustration/database_er.mmd`` (Study, Dataset, Datasheet, Scaffold).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .converter import DataConverter
from .db.repository import CatalogRepository
from .db.schemas import (
    DatasetRead,
    DatasheetRead,
    ScaffoldRead,
    StatisticsRead,
    StudyRead,
)
from .db.session import get_session
from .loaders import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from .catalog.initialize import initialize_database

    initialize_database(
        force_export=_env_flag("PE_DB_FORCE_EXPORT"),
        force_standardize=_env_flag("PE_DB_FORCE_STANDARDIZE"),
    )
    yield


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

converter = DataConverter()
loader = DataLoader()


def _legacy_model_to_study(source_model: str) -> str:
    legacy_map = {
        "dp": "deepprime",
        "dp_ft": "deepprime",
        "pd1": "pridict1",
        "pd2": "pridict2",
    }
    return legacy_map.get(source_model.strip().lower(), source_model.strip().lower())


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
            "data": "/api/data",
            "statistics": "/api/statistics",
            "export": "POST /api/export",
            "health": "/health",
        },
    }


@app.get("/api/studies", response_model=list[StudyRead])
async def list_studies():
    """List all studies in the catalog."""
    with get_session() as session:
        return CatalogRepository(session).list_studies()


@app.get("/api/scaffolds", response_model=list[ScaffoldRead])
async def list_scaffolds():
    """List pegRNA scaffold definitions (id, name, sequence)."""
    with get_session() as session:
        return CatalogRepository(session).list_scaffolds()


@app.get("/api/scaffolds/{scaffold_id}", response_model=ScaffoldRead)
async def get_scaffold(scaffold_id: int):
    with get_session() as session:
        scaffold = CatalogRepository(session).get_scaffold(scaffold_id)
    if scaffold is None:
        raise HTTPException(status_code=404, detail=f"Scaffold not found: {scaffold_id}")
    return scaffold


@app.get("/api/datasets", response_model=list[DatasetRead])
async def list_catalog_datasets(
    study: Optional[str] = Query(None, description="Filter by study key (e.g. deepprime)"),
):
    """List datasets registered in the catalog."""
    with get_session() as session:
        return CatalogRepository(session).list_datasets(study_name=study)


@app.get("/api/datasheets", response_model=list[DatasheetRead])
async def list_datasheets(
    study: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
):
    """List datasheets (cell line × PE system) with scaffold and file metadata."""
    with get_session() as session:
        return CatalogRepository(session).list_datasheets(
            study_name=study,
            dataset_name=dataset,
        )


@app.get("/api/data")
async def get_data(
    study: Optional[str] = Query(
        None,
        description="Study key (e.g., deepprime, pridict1, pridict2, minsepie).",
    ),
    dataset: str = Query(
        ...,
        description="Dataset within the study (e.g., deepprime-clinvar, library1).",
    ),
    cell_line: str = Query(..., description="Cell line (e.g., HEK293T, A549, DLD1)"),
    pe_system: str = Query(..., description="PE system (e.g., PE2, PE2max, PE4max)"),
    source_model: Optional[str] = Query(
        None,
        description="Deprecated alias for study (dp, dp_ft, pd1, pd2, etc.).",
    ),
    target_format: Literal["std", "oped", "deepprime", "pridict", "pridict2"] = Query(
        "std",
        description="Target format for the data",
    ),
    limit: Optional[int] = Query(None, description="Limit number of records returned"),
):
    try:
        resolved_study = study
        if not resolved_study and source_model:
            resolved_study = _legacy_model_to_study(source_model)
        if not resolved_study:
            raise HTTPException(
                status_code=422,
                detail="Missing required query parameter: study (or deprecated source_model).",
            )

        catalog_meta = None
        with get_session() as session:
            catalog_meta = CatalogRepository(session).find_datasheet(
                study_name=resolved_study,
                dataset_name=dataset,
                cell_line=cell_line,
                pe_system=pe_system,
            )

        data = loader.load_data(
            study=resolved_study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            target_format=target_format,
        )

        if limit is not None and limit > 0:
            data = data.head(limit)

        metadata = {
            "study": resolved_study,
            "dataset": dataset,
            "cell_line": cell_line,
            "pe_system": pe_system,
            "format": target_format,
            "total_records": len(data),
            "columns": list(data.columns),
        }
        if catalog_meta is not None:
            metadata["datasheet_id"] = catalog_meta.id
            metadata["scaffold_id"] = catalog_meta.scaffold_id
            if catalog_meta.scaffold is not None:
                metadata["scaffold_sequence"] = catalog_meta.scaffold.sequence

        return {
            "status": "success",
            "metadata": metadata,
            "data": data.to_dict(orient="records"),
        }

    except FileNotFoundError as exc:
        logger.error("Data file not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error loading data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error loading data: {exc}") from exc


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
        with get_session() as session:
            return CatalogRepository(session).compute_statistics(
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


@app.post("/api/export")
async def export_data(
    study: Optional[str] = Query(
        None,
        description="Study key to export (e.g. deepprime). Omit to export all supported studies.",
    ),
    force_reexport: bool = Query(
        False,
        description="Re-export even when ``datasets/exported/{study}`` already exists.",
    ),
    standardize: bool = Query(
        True,
        description="Also standardize exported CSVs to ``datasets/standardized/`` parquet.",
    ),
    force_standardize: bool = Query(
        False,
        description="Re-standardize even when parquet output already exists.",
    ),
):
    """Export raw study files (and optionally standardize). Refreshes Datasheet catalog rows."""
    try:
        if study is None and not force_reexport and not force_standardize and standardize:
            converter.initialize_database(
                force_export=force_reexport,
                force_standardize=force_standardize,
            )
        else:
            converter.export_raw(study=study, force_reexport=force_reexport)
            if standardize:
                from .utils.standardize_data import standardize_exported_data

                standardize_exported_data(study=study, force=force_standardize)
        with get_session() as session:
            count = len(CatalogRepository(session).list_datasheets())
        return {
            "status": "success",
            "study": study or "all",
            "force_reexport": force_reexport,
            "standardized": standardize,
            "datasheets_in_catalog": count,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/convert")
async def convert_data(
    study: str = Query(..., description="Study key (e.g., deepprime, pridict1, pridict2)"),
    dataset: str = Query(..., description="Dataset in the selected study"),
    cell_line: str = Query(..., description="Cell line name"),
    pe_system: str = Query(..., description="PE system"),
):
    try:
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
    except FileNotFoundError as exc:
        logger.error("Source file not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error converting data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error converting data: {exc}") from exc


@app.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "catalog_database": str(settings.catalog_db_path),
        "catalog_database_exists": settings.catalog_db_path.exists(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

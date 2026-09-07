# FastAPI endpoints for PE Ensemble service
import os

# Avoid joblib/loky spawning extra process pools (PRIDICT2 imports sklearn).
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional, Dict, Any, Union
import logging

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models.model_factory import ModelFactory
from .models import weights_registry
from pe_ensemble import library
from pe_ensemble.library import (
    EnsembleRequest,
    EvaluationRequest,
    PeEnsembleLibraryError,
    SplitQueryParams,
    TrainingRequest,
    TuningRequest,
    build_pe_db_filter_params as _build_pe_db_filter_params,
    is_supported_model,
)
from .evaluation.schemas import (
    EvaluationJobCreatedResponse,
    EvaluationLogResponse,
)
from .ensemble.schemas import (
    EnsembleJobCreatedResponse,
    EnsembleLogResponse,
)
from .training.tuning_schemas import (
    TuningJobCreatedResponse,
    TuningLogResponse,
)
from pe_common.filter_params import CatalogFilterQuery, SplitExportQuery, query_dependency
from pe_common.devices import list_devices as list_compute_devices
from pe_common.devices import default_device_id, resolve_device
from .training.schemas import (
    TrainingJobCreatedResponse,
    TrainingLogResponse,
)

logger = logging.getLogger(__name__)

_catalog_filters = query_dependency(CatalogFilterQuery)
_split_export = query_dependency(SplitExportQuery)

GRACEFUL_SHUTDOWN_SECONDS = 5


def _shutdown_joblib_loky() -> None:
    """Release joblib/loky semaphores created when sklearn is imported."""
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pe-ensemble-sync")
    loop.set_default_executor(executor)
    try:
        loaded = library.load_active_plugins()
        if loaded:
            logger.info("Loaded PE Ensemble plugins: %s", ", ".join(loaded))
        yield
    finally:
        from .plugins.scheduler import shutdown_validation_scheduler

        library.shutdown_compute()
        shutdown_validation_scheduler()
        _shutdown_joblib_loky()
        executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(
    title="PE Ensemble API",
    description="Unified API for training and evaluating Prime Editing prediction models (DeepPrime, OPED, PRIDICT2)",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS(Cross-Origin Resource Sharing, allow all origins for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PE_DB_URL = os.getenv("PE_DB_URL", "http://localhost:8000")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/vendor/models")
DEEPPRIME_PATH = os.getenv("DEEPPRIME_PATH", f"{MODEL_PATH}/deepprime")
OPED_PATH = os.getenv("OPED_PATH", f"{MODEL_PATH}/oped")
PRIDICT2_PATH = os.getenv("PRIDICT2_PATH", f"{MODEL_PATH}/pridict2")
WEIGHTS_ROOT = os.getenv("WEIGHTS_ROOT", str(weights_registry.weights_root()))

# Store loaded models in memory
_loaded_models: Dict[str, Any] = {}

class PredictionRequest(BaseModel):
    model_name: str
    sequences: List[str]
    cell_type: Optional[str] = None
    weights: Optional[str] = None
    device: Optional[str] = "auto"


def _default_weight_id_for_model(model_name: str, model: Any) -> Optional[str]:
    if model_name == "deepprime":
        from deepprime.models.load_model import load_deepprime

        _, model_type = load_deepprime(
            getattr(model, "pe_system", "PE2max"),
            getattr(model, "cell_type", "HEK293T"),
            silent=True,
        )
        return model_type
    if model_name == "oped":
        return "pegRNA_Model_Merged_saved.order3_decoder_weights"
    return None


def _request_pe_db_filtered(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return library.filter_pe_db(params)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch data from PE-DB: {exc}"
        ) from exc


def _http_get_job(kind: library.JobKind, job_id: str) -> Dict[str, Any]:
    try:
        return library.get_job(kind, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _http_queue_job(kind: library.JobKind, request: Any, response_cls):
    try:
        job_id, manifest = library.queue_job(kind, request)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response_cls(
        job_id=job_id,
        status=manifest["status"],
        message=library.queued_message(kind, manifest),
    )


def _http_delete_job(kind: library.JobKind, job_id: str) -> Dict[str, Any]:
    _http_get_job(kind, job_id)
    manifest = library.begin_kill(kind, job_id)
    asyncio.create_task(asyncio.to_thread(library.finalize_kill, kind, job_id))
    return {
        "job_id": job_id,
        "accepted": True,
        "status": manifest.get("status") if manifest else "deleted",
    }


def _http_job_logs(kind: library.JobKind, job_id: str, offset: int, response_cls):
    try:
        payload = library.job_logs(kind, job_id, offset=offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return response_cls(**payload)


def _http_device_queues() -> Dict[str, Any]:
    return {
        "default": default_device_id(),
        "devices": library.device_snapshot(),
    }


@app.get("/")
async def root():
    """
    Root endpoint providing service info
    Returns:
    {
        "service": "PE Ensemble",
        "version": "0.2.0",
        "status": "running",
        "pe_db_url": PE_DB_URL,
        "model_paths": {
            "deepprime": DEEPPRIME_PATH,
            "oped": OPED_PATH,
            "pridict2": PRIDICT2_PATH
        }
    """
    return {
        "service": "PE Ensemble",
        "version": "0.2.0",
        "status": "running",
        "pe_db_url": PE_DB_URL,
        "data_filter": "/data/filter",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/data/filter")
async def export_filtered_data(
    filters: Annotated[CatalogFilterQuery, Depends(_catalog_filters)],
    split: Annotated[SplitExportQuery, Depends(_split_export)],
    format_: str = Query(
        ...,
        alias="format",
        description="Output format (same as PE-DB GET /api/filter).",
    ),
):
    """Proxy PE-DB ``GET /api/filter`` (PE Hub ``exportFiltered`` contract).

    Returns converted, standardizable data grouped by datasheet. Use this from
    PE Ensemble clients when you need the raw export payload without running
    model evaluation.
    """
    split_data = split.model_dump()
    if split_data.get("split_strategy") is None:
        split_data["split_strategy"] = "none"
    params = _build_pe_db_filter_params(
        model_format=format_,
        split=SplitQueryParams(**split_data),
        **filters.model_dump(),
    )
    return await asyncio.to_thread(_request_pe_db_filtered, params)


@app.get("/models")
async def list_models():
    """List all available models"""
    models = library.list_model_catalog()
    return {"models": models, "count": len(models)}


@app.get("/models/{model_name}/training-presets")
async def get_training_presets(
    model_name: str,
    study: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
    cell_line: Optional[str] = Query(None),
    pe_system: Optional[str] = Query(None),
    hyperparameter_mode: Literal["merge", "replace"] = Query("merge"),
):
    """Resolve training hyperparameters for a model and dataset filter."""
    try:
        return library.resolve_training_presets(
            model_name,
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            hyperparameter_mode=hyperparameter_mode,
        )
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/models/{model_name}/weights")
async def list_model_weights(model_name: str):
    """List registered weight sets available for a model."""
    try:
        entries = library.list_weight_entries(model_name)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"model": model_name.strip().lower(), "weights": entries, "count": len(entries)}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get predictions from a model"""
    if not is_supported_model(request.model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    model_kwargs: Dict[str, Any] = {}
    if request.cell_type:
        model_kwargs["cell_type"] = request.cell_type

    weight_id: Optional[str] = request.weights
    try:
        model = ModelFactory.create_model(
            request.model_name,
            device=resolve_device(request.device),
            **model_kwargs,
        )
        if not weight_id:
            weight_id = _default_weight_id_for_model(request.model_name, model)
        if weight_id:
            model.load_weights_by_name(weight_id)
        else:
            model.load_model()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "model": request.model_name,
        "weights": weight_id,
        "predictions": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Prediction endpoint - implementation pending",
    }


@app.get("/devices")
async def list_devices():
    """List compute devices available for training and inference."""
    devices = list_compute_devices()
    return {
        "default": default_device_id(),
        "devices": [device.to_dict() for device in devices],
        "count": len(devices),
    }


@app.get("/train/devices")
async def training_device_status():
    """Per-device occupancy and queue depth for training jobs."""
    return _http_device_queues()


@app.post("/train")
async def train_model(request: TrainingRequest):
    """Queue an asynchronous model training job."""
    try:
        library.require_supported_model(request.model_name)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _http_queue_job("train", request, TrainingJobCreatedResponse)


@app.get("/train/jobs")
async def list_training_jobs(limit: int = Query(50, ge=1, le=200)):
    """List recent training jobs (newest first)."""
    jobs = library.list_job_summaries("train", limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Return training job status and result metadata."""
    _http_get_job("train", job_id)
    return library.job_status("train", job_id)


@app.delete("/train/jobs/{job_id}", status_code=202)
async def delete_training_job(job_id: str):
    """Stop a queued or running training job and remove its on-disk artifacts."""
    return _http_delete_job("train", job_id)


@app.get("/train/logs/{job_id}")
async def get_training_logs(
    job_id: str,
    offset: int = Query(0, ge=0, description="Byte offset into the log file"),
):
    """Return incremental training log output for a job."""
    return _http_job_logs("train", job_id, offset, TrainingLogResponse)


@app.post("/tune")
async def tune_model(request: TuningRequest):
    """Queue an asynchronous hyperparameter tuning job."""
    try:
        library.require_supported_model(request.training.model_name)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _http_queue_job("tune", request, TuningJobCreatedResponse)


@app.get("/tune/jobs")
async def list_tuning_jobs(limit: int = Query(50, ge=1, le=200)):
    jobs = library.list_job_summaries("tune", limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/tune/status/{job_id}")
async def get_tuning_status(job_id: str):
    _http_get_job("tune", job_id)
    return library.job_status("tune", job_id)


@app.delete("/tune/jobs/{job_id}", status_code=202)
async def delete_tuning_job(job_id: str):
    return _http_delete_job("tune", job_id)


@app.get("/tune/logs/{job_id}")
async def get_tuning_logs(
    job_id: str,
    offset: int = Query(0, ge=0, description="Byte offset into the log file"),
):
    return _http_job_logs("tune", job_id, offset, TuningLogResponse)


@app.get("/tune/devices")
async def tuning_device_status():
    return _http_device_queues()


@app.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Queue an asynchronous benchmark / evaluation job."""
    try:
        request = library.resolve_evaluation(request)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _http_queue_job("evaluate", request, EvaluationJobCreatedResponse)


@app.get("/evaluate/jobs")
async def list_evaluation_jobs(limit: int = Query(50, ge=1, le=200)):
    jobs = library.list_job_summaries("evaluate", limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/evaluate/status/{job_id}")
async def get_evaluation_status(job_id: str):
    _http_get_job("evaluate", job_id)
    return library.job_status("evaluate", job_id)


@app.delete("/evaluate/jobs/{job_id}", status_code=202)
async def delete_evaluation_job(job_id: str):
    """Stop a queued or running evaluation job and remove its on-disk artifacts."""
    return _http_delete_job("evaluate", job_id)


@app.get("/evaluate/logs/{job_id}")
async def get_evaluation_logs(
    job_id: str,
    offset: int = Query(0, ge=0),
):
    return _http_job_logs("evaluate", job_id, offset, EvaluationLogResponse)


@app.get("/evaluate/devices")
async def evaluation_device_status():
    return _http_device_queues()


@app.get("/ensemble/methods")
async def list_ensemble_combine_methods():
    """List supported no-retrain prediction fusion methods."""
    methods = library.combine_method_help()
    return {"methods": methods, "count": len(methods)}


@app.post("/ensemble")
async def run_ensemble(request: EnsembleRequest):
    """Queue an asynchronous ensemble evaluation job."""
    try:
        library.validate_ensemble_request(request)
    except PeEnsembleLibraryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _http_queue_job("ensemble", request, EnsembleJobCreatedResponse)


@app.get("/ensemble/jobs")
async def list_ensemble_evaluation_jobs(limit: int = Query(50, ge=1, le=200)):
    jobs = library.list_job_summaries("ensemble", limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/ensemble/status/{job_id}")
async def get_ensemble_status(job_id: str):
    _http_get_job("ensemble", job_id)
    return library.job_status("ensemble", job_id)


@app.delete("/ensemble/jobs/{job_id}", status_code=202)
async def delete_ensemble_job(job_id: str):
    """Stop a queued or running ensemble job and remove its on-disk artifacts."""
    return _http_delete_job("ensemble", job_id)


@app.get("/ensemble/logs/{job_id}")
async def get_ensemble_logs(
    job_id: str,
    offset: int = Query(0, ge=0),
):
    return _http_job_logs("ensemble", job_id, offset, EnsembleLogResponse)


@app.get("/ensemble/devices")
async def ensemble_device_status():
    return _http_device_queues()


@app.get("/models/plugins")
async def list_plugin_bundles():
    from .plugins.manager import list_plugins

    plugins = await asyncio.to_thread(list_plugins)
    return {"plugins": plugins, "count": len(plugins)}


@app.get("/models/plugins/{name}")
async def get_plugin_bundle(name: str):
    from .plugins.manager import get_plugin

    try:
        return await asyncio.to_thread(get_plugin, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/plugins/{name}/validation.log")
async def get_plugin_validation_log(
    name: str,
    offset: int = Query(0, ge=0),
):
    from .plugins.manager import read_validation_log

    try:
        log_chunk, next_offset = await asyncio.to_thread(read_validation_log, name, offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "name": name.strip().lower(),
        "offset": offset,
        "next_offset": next_offset,
        "log": log_chunk,
    }


@app.post("/models/plugins")
async def upload_plugin_bundle(
    name: Optional[str] = Form(None),
    version: Optional[str] = Form("0.1.0"),
    display_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    wrapper_class: Optional[str] = Form(None),
    weight_format: Optional[str] = Form(None),
    authors: Optional[str] = Form(None),
    convert_entrypoint: str = Form("convert"),
    pe_db_format: Optional[str] = Form(None),
    output_columns: Optional[str] = Form(None),
    required_std_columns: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    hyperparameters_json: Optional[str] = Form(None),
    weights_json: Optional[str] = Form(None),
    replace_existing: bool = Form(False),
    convert_file: Optional[UploadFile] = File(None),
    wrapper_file: Optional[UploadFile] = File(None),
    bundle_zip: Optional[UploadFile] = File(None),
    manifest_file: Optional[UploadFile] = File(None),
    weight_id: Optional[str] = Form(None),
    weight_file: Optional[UploadFile] = File(None),
):
    from pe_common.plugins import PluginError

    from .plugins.manager import upload_plugin_bundle

    convert_bytes: Optional[bytes] = None
    wrapper_bytes: Optional[bytes] = None
    bundle_zip_bytes: Optional[bytes] = None
    manifest_bytes: Optional[bytes] = None
    weight_uploads: Optional[list] = None
    if bundle_zip is not None:
        bundle_zip_bytes = await bundle_zip.read()
    if manifest_file is not None:
        manifest_bytes = await manifest_file.read()
    if convert_file is not None:
        convert_bytes = await convert_file.read()
    if wrapper_file is not None:
        wrapper_bytes = await wrapper_file.read()
    if weight_file is not None and weight_id and weight_id.strip():
        weight_uploads = [(weight_id.strip(), await weight_file.read())]

    try:
        result = await asyncio.to_thread(
            upload_plugin_bundle,
            name=name or None,
            version=version or "0.1.0",
            display_name=display_name or "",
            description=description or "",
            authors=authors,
            wrapper_class=wrapper_class or "",
            convert_entrypoint=convert_entrypoint,
            pe_db_format=pe_db_format,
            weight_format=weight_format or "",
            output_columns=output_columns,
            required_std_columns=required_std_columns,
            label_column=label_column,
            hyperparameters_json=hyperparameters_json,
            weights_json=weights_json,
            convert_bytes=convert_bytes,
            wrapper_bytes=wrapper_bytes,
            bundle_zip_bytes=bundle_zip_bytes,
            manifest_bytes=manifest_bytes,
            weight_uploads=weight_uploads,
            replace_existing=replace_existing,
        )
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.post("/models/plugins/{name}/validate", status_code=202)
async def validate_plugin_bundle(name: str):
    from pe_common.plugins import PluginError

    from .plugins.manager import queue_validation

    try:
        created = await asyncio.to_thread(queue_validation, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PluginError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    from .plugins.schemas import PluginValidationJobCreatedResponse

    return PluginValidationJobCreatedResponse(
        job_id=created["job_id"],
        plugin_name=created["plugin_name"],
        status=created["status"],
        message=created["message"],
    )


@app.get("/models/plugins/{name}/validate/status/{job_id}")
async def get_plugin_validation_status(name: str, job_id: str):
    from .plugins.schemas import job_summary
    from .plugins.validation_jobs import get_job

    try:
        manifest = await asyncio.to_thread(get_job, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if manifest.get("plugin_name") != name.strip().lower():
        raise HTTPException(status_code=404, detail="Validation job not found for this plugin")

    summary = job_summary(manifest).model_dump()
    if manifest.get("result") is not None:
        summary["result"] = manifest["result"]
    return summary


@app.get("/models/plugins/{name}/validate/logs/{job_id}")
async def get_plugin_validation_job_logs(
    name: str,
    job_id: str,
    offset: int = Query(0, ge=0),
):
    from .plugins.schemas import PluginValidationLogResponse
    from .plugins.validation_jobs import get_job, read_logs

    try:
        manifest = await asyncio.to_thread(get_job, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if manifest.get("plugin_name") != name.strip().lower():
        raise HTTPException(status_code=404, detail="Validation job not found for this plugin")

    log_chunk, next_offset = await asyncio.to_thread(read_logs, job_id, offset=offset)
    return PluginValidationLogResponse(
        job_id=job_id,
        plugin_name=manifest["plugin_name"],
        status=manifest["status"],
        offset=offset,
        next_offset=next_offset,
        log=log_chunk,
    )


@app.delete("/models/plugins/{name}/validate/jobs/{job_id}", status_code=202)
async def cancel_plugin_validation_job(name: str, job_id: str):
    from .plugins.scheduler import get_validation_scheduler
    from .plugins.validation_jobs import get_job

    try:
        manifest = await asyncio.to_thread(get_job, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if manifest.get("plugin_name") != name.strip().lower():
        raise HTTPException(status_code=404, detail="Validation job not found for this plugin")

    accepted = await asyncio.to_thread(get_validation_scheduler().cancel, job_id)
    return {
        "job_id": job_id,
        "accepted": accepted,
        "status": manifest.get("status"),
    }


@app.post("/models/plugins/{name}/activate")
async def activate_plugin_bundle_endpoint(name: str):
    from pe_common.plugins import PluginError

    from .plugins.manager import activate_plugin_bundle

    try:
        return await asyncio.to_thread(activate_plugin_bundle, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PluginError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/models/plugins/{name}")
async def delete_plugin_bundle_endpoint(name: str):
    from .plugins.manager import delete_plugin_bundle

    try:
        return await asyncio.to_thread(delete_plugin_bundle, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )

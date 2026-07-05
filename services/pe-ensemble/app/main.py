# FastAPI endpoints for PE Ensemble service
import os

# Avoid joblib/loky spawning extra process pools (PRIDICT2 imports sklearn).
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict, Any, Union
import logging

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models.model_factory import ModelFactory
from .models import weights_registry
from .models.registry import model_registry
from .training.config import is_supported_model
from .training.data import (
    build_pe_db_filter_params as _build_pe_db_filter_params,
    request_pe_db_filtered,
)
from .compute.device_scheduler import get_scheduler
from .compute.job_lifecycle import begin_job_kill, finalize_job_kill
from .evaluation.jobs import (
    create_job as create_eval_job,
    delete_job as delete_eval_job,
    get_job as get_eval_job,
    job_summary as eval_job_summary,
    list_jobs as list_eval_jobs,
    read_logs as read_eval_logs,
)
from .evaluation.schemas import (
    EvaluationJobCreatedResponse,
    EvaluationLogResponse,
    EvaluationRequest,
)
from .training.jobs import create_job, delete_job as delete_train_job, get_job, job_summary, list_jobs, read_logs
from pe_common.devices import list_devices as list_compute_devices
from pe_common.devices import default_device_id, resolve_device
from .training.schemas import (
    SplitQueryParams,
    SplitStrategy,
    TrainingJobCreatedResponse,
    TrainingJobSummary,
    TrainingLogResponse,
    TrainingRequest,
    default_training_split as _default_training_split,
)

logger = logging.getLogger(__name__)

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
        from .plugin_loader import load_active_plugins

        loaded = load_active_plugins()
        if loaded:
            logger.info("Loaded PE Ensemble plugins: %s", ", ".join(loaded))
        yield
    finally:
        from .compute.device_scheduler import shutdown_scheduler
        from .plugins.scheduler import shutdown_validation_scheduler

        shutdown_scheduler()
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
        return request_pe_db_filtered(params)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch data from PE-DB: {exc}"
        ) from exc


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
    format_: str = Query(
        ...,
        alias="format",
        description="Output format (same as PE-DB GET /api/filter).",
    ),
    study: Optional[List[str]] = Query(None, description="Filter by study key."),
    dataset: Optional[List[str]] = Query(None, description="Filter by dataset name."),
    cell_line: Optional[List[str]] = Query(None, description="Filter by cell line."),
    pe_system: Optional[List[str]] = Query(None, description="Filter by PE system."),
    edit_type: Optional[List[str]] = Query(None, description="Filter edits by type."),
    edit_length: Optional[List[int]] = Query(None, description="Filter edits by length."),
    edit_efficiency_min: Optional[float] = Query(None, description="Minimum editing efficiency."),
    edit_efficiency_max: Optional[float] = Query(None, description="Maximum editing efficiency."),
    edit_scope: Optional[List[str]] = Query(None, description="Filter by edit scope."),
    experimental_method: Optional[List[str]] = Query(None, description="Filter by experimental method."),
    target_context: Optional[List[str]] = Query(None, description="Filter by target context."),
    scaffold_name: Optional[List[str]] = Query(None, description="Filter by pegRNA scaffold name."),
    split_strategy: SplitStrategy = Query(
        "none",
        description="Required by PE-DB when exporting formatted data.",
    ),
    train_pct: Optional[float] = Query(None, ge=0.0, le=1.0),
    val_pct: Optional[float] = Query(None, ge=0.0, le=1.0),
    test_pct: Optional[float] = Query(None, ge=0.0, le=1.0),
    cv_folds: Optional[int] = Query(None, ge=2),
    use_original_fold: bool = Query(False),
    original_fold_test_value: float = Query(-1.0),
    split_random_state: int = Query(42, ge=0),
    merge: bool = Query(False),
):
    """Proxy PE-DB ``GET /api/filter`` (PE Hub ``exportFiltered`` contract).

    Returns converted, standardizable data grouped by datasheet. Use this from
    PE Ensemble clients when you need the raw export payload without running
    model evaluation.
    """
    params = _build_pe_db_filter_params(
        model_format=format_,
        split=SplitQueryParams(
            split_strategy=split_strategy,
            train_pct=train_pct,
            val_pct=val_pct,
            test_pct=test_pct,
            cv_folds=cv_folds,
            use_original_fold=use_original_fold,
            original_fold_test_value=original_fold_test_value,
            split_random_state=split_random_state,
            merge=merge,
        ),
        study=study,
        dataset=dataset,
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
    )
    return await asyncio.to_thread(_request_pe_db_filtered, params)


@app.get("/models")
async def list_models():
    """List all available models"""
    models = model_registry.list_catalog_entries()
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
    model_name = model_name.strip().lower()
    if not is_supported_model(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    from .training.hyperparameter_presets import resolve_hyperparameters

    resolved = resolve_hyperparameters(
        model_name,
        study=study,
        dataset=dataset,
        cell_line=cell_line,
        pe_system=pe_system,
        user_overrides=None,
        mode=hyperparameter_mode,
    )
    return {
        "model": model_name,
        "preset_key": resolved.preset_key,
        "preset_source": resolved.preset_source,
        "hyperparameter_mode": hyperparameter_mode,
        "hyperparameters": resolved.hyperparameters,
    }


@app.get("/models/{model_name}/weights")
async def list_model_weights(model_name: str):
    """List registered weight sets available for a model."""
    model_name = model_name.strip().lower()
    if not is_supported_model(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    entries = model_registry.list_weight_entries(model_name)
    return {"model": model_name, "weights": entries, "count": len(entries)}


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
    return {
        "default": default_device_id(),
        "devices": get_scheduler().device_snapshot(),
    }


@app.post("/train")
async def train_model(request: TrainingRequest):
    """Queue an asynchronous model training job."""
    model_name = request.model_name.strip().lower()
    if not is_supported_model(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    job_id = create_job(request)
    get_scheduler().submit_training(job_id, request)
    manifest = get_job(job_id)
    message = "Training job started"
    if manifest.get("queue_position"):
        message = f"Training job queued (position {manifest['queue_position']})"
    return TrainingJobCreatedResponse(
        job_id=job_id,
        status=manifest["status"],
        message=message,
    )


@app.get("/train/jobs")
async def list_training_jobs(limit: int = Query(50, ge=1, le=200)):
    """List recent training jobs (newest first)."""
    manifests = list_jobs(limit=limit)
    return {
        "jobs": [job_summary(m).model_dump() for m in manifests],
        "count": len(manifests),
    }


@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str):
    """Return training job status and result metadata."""
    try:
        manifest = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    summary = job_summary(manifest).model_dump()
    if manifest.get("result") is not None:
        summary["result"] = manifest["result"]
    return summary


@app.delete("/train/jobs/{job_id}", status_code=202)
async def delete_training_job(job_id: str):
    """Stop a queued or running training job and remove its on-disk artifacts."""
    try:
        get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manifest = begin_job_kill("train", job_id, get_job=get_job)
    asyncio.create_task(
        asyncio.to_thread(
            finalize_job_kill,
            "train",
            job_id,
            get_job=get_job,
            delete_job=delete_train_job,
        )
    )
    return {
        "job_id": job_id,
        "accepted": True,
        "status": manifest.get("status") if manifest else "deleted",
    }


@app.get("/train/logs/{job_id}")
async def get_training_logs(
    job_id: str,
    offset: int = Query(0, ge=0, description="Byte offset into the log file"),
):
    """Return incremental training log output for a job."""
    try:
        manifest = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log_chunk, next_offset = read_logs(job_id, offset=offset)
    return TrainingLogResponse(
        job_id=job_id,
        status=manifest["status"],
        offset=offset,
        next_offset=next_offset,
        log=log_chunk,
    )


@app.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Queue an asynchronous benchmark / evaluation job."""
    model_name = request.model_name.strip().lower()
    if not is_supported_model(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    weight_id = request.weights.strip()
    if not weight_id:
        raise HTTPException(status_code=400, detail="weights is required")
    try:
        model_registry.validate_weight_selection(model_name, weight_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = create_eval_job(request)
    get_scheduler().submit_evaluation(job_id, request)
    manifest = get_eval_job(job_id)
    message = "Evaluation job started"
    if manifest.get("queue_position"):
        message = f"Evaluation job queued (position {manifest['queue_position']})"
    return EvaluationJobCreatedResponse(
        job_id=job_id,
        status=manifest["status"],
        message=message,
    )


@app.get("/evaluate/jobs")
async def list_evaluation_jobs(limit: int = Query(50, ge=1, le=200)):
    manifests = list_eval_jobs(limit=limit)
    return {
        "jobs": [eval_job_summary(m).model_dump() for m in manifests],
        "count": len(manifests),
    }


@app.get("/evaluate/status/{job_id}")
async def get_evaluation_status(job_id: str):
    try:
        manifest = get_eval_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    summary = eval_job_summary(manifest).model_dump()
    if manifest.get("result") is not None:
        summary["result"] = manifest["result"]
    return summary


@app.delete("/evaluate/jobs/{job_id}", status_code=202)
async def delete_evaluation_job(job_id: str):
    """Stop a queued or running evaluation job and remove its on-disk artifacts."""
    try:
        get_eval_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manifest = begin_job_kill("evaluate", job_id, get_job=get_eval_job)
    asyncio.create_task(
        asyncio.to_thread(
            finalize_job_kill,
            "evaluate",
            job_id,
            get_job=get_eval_job,
            delete_job=delete_eval_job,
        )
    )
    return {
        "job_id": job_id,
        "accepted": True,
        "status": manifest.get("status") if manifest else "deleted",
    }


@app.get("/evaluate/logs/{job_id}")
async def get_evaluation_logs(
    job_id: str,
    offset: int = Query(0, ge=0),
):
    try:
        manifest = get_eval_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    log_chunk, next_offset = read_eval_logs(job_id, offset=offset)
    return EvaluationLogResponse(
        job_id=job_id,
        status=manifest["status"],
        offset=offset,
        next_offset=next_offset,
        log=log_chunk,
    )


@app.get("/evaluate/devices")
async def evaluation_device_status():
    return {
        "default": default_device_id(),
        "devices": get_scheduler().device_snapshot(),
    }


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

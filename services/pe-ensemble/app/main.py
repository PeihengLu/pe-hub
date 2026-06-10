# FastAPI endpoints for PE Ensemble service
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict, Any, Union
import logging
import os

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models.model_factory import ModelFactory
from .models import weights_registry
from .training.config import MODEL_FORMAT, SUPPORTED_MODELS
from .training.data import (
    build_pe_db_filter_params as _build_pe_db_filter_params,
    request_pe_db_filtered,
)
from .compute.device_scheduler import get_scheduler
from .compute.job_lifecycle import kill_and_remove_job
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pe-ensemble-sync")
    loop.set_default_executor(executor)
    try:
        yield
    finally:
        from .compute.device_scheduler import shutdown_scheduler

        shutdown_scheduler()
        executor.shutdown(wait=False, cancel_futures=True)


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
    format_: Literal["std", "oped", "deepprime", "pridict", "pridict2"] = Query(
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
    models = [
        {
            "name": "deepprime",
            "description": "CNN model for PE efficiency prediction",
            "type": "neural_network",
            "status": "available"
        },
        # {
        #     "name": "pridict",
        #     "description": "RNN based model for PE efficiency prediction",
        #     "type": "pssm",
        #     "status": "available"
        # },
        {
            "name": "pridict2",
            "description": "Improved version of PRIDICT with transfer learning",
            "type": "pssm",
            "status": "available"
        },
        {
            "name": "oped",
            "description": "Optimized Prime Editor prediction model using transformer architecture",
            "type": "neural_network",
            "status": "available"
        }
    ]
    
    return {"models": models, "count": len(models)}


@app.get("/models/{model_name}/weights")
async def list_model_weights(model_name: str):
    """List registered weight sets available for a model."""
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")
    if model_name == "pridict2":
        from .models.pridict2_wrapper import PRIDICT2ModelWrapper

        entries = PRIDICT2ModelWrapper.list_available_weight_entries()
        return {"model": model_name, "weights": entries, "count": len(entries)}

    entries = weights_registry.list_entries(model_name)
    return {"model": model_name, "weights": entries, "count": len(entries)}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get predictions from a model"""
    if request.model_name not in ["deepprime", "pridict", "pridict2", "oped"]:
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
    if model_name not in SUPPORTED_MODELS:
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


@app.delete("/train/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Stop a queued or running training job and remove its on-disk artifacts."""
    try:
        get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    kill_and_remove_job(
        "train",
        job_id,
        get_job=get_job,
        delete_job=delete_train_job,
    )
    return {"job_id": job_id, "deleted": True}


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
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    weight_id = request.weights.strip()
    if not weight_id:
        raise HTTPException(status_code=400, detail="weights is required")
    try:
        if model_name == "pridict2":
            from .models.pridict2_wrapper import PRIDICT2ModelWrapper

            PRIDICT2ModelWrapper.resolve_weight_selection(weight_id)
        else:
            weights_registry.resolve_dir(model_name, weight_id)
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


@app.delete("/evaluate/jobs/{job_id}")
async def delete_evaluation_job(job_id: str):
    """Stop a queued or running evaluation job and remove its on-disk artifacts."""
    try:
        get_eval_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    kill_and_remove_job(
        "evaluate",
        job_id,
        get_job=get_eval_job,
        delete_job=delete_eval_job,
    )
    return {"job_id": job_id, "deleted": True}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )

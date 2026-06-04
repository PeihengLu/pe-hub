# FastAPI endpoints for PE Ensemble service
from typing import List, Optional, Dict, Any
import logging
import os

import pandas as pd
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models.model_factory import ModelFactory
from pe_common.constants import DEVICE

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PE Ensemble API",
    description="Unified API for training and evaluating Prime Editing prediction models (DeepPrime, OPED, PRIDICT2)",
    version="0.2.0"
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

# Store loaded models in memory
_loaded_models: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    model_name: str
    sequences: List[str]
    cell_type: Optional[str] = None


class TrainingRequest(BaseModel):
    model_name: str
    dataset_source: str
    dataset_name: str
    hyperparameters: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    model_name: str
    # Optional bundled weight set to evaluate (see GET /models/{name}/weights).
    weights: Optional[str] = None
    # Data source in PE-DB (used when `records` is not provided).
    study: Optional[str] = None
    dataset: Optional[str] = None
    cell_line: Optional[str] = None
    pe_system: Optional[str] = None
    # Inline records, already in the model's native format (bypasses the PE-DB
    # fetch when provided).
    records: Optional[List[Dict[str, Any]]] = None
    # Model constructor kwargs (e.g. pe_system/cell_type for DeepPrime,
    # model_name for PRIDICT2).
    model_kwargs: Optional[Dict[str, Any]] = None


SUPPORTED_MODELS = ["deepprime", "pridict2", "oped"]

# Map each model to the PE-DB conversion format it consumes. PE-DB owns
# standardized -> model-format conversion (GET /api/filter?format=...).
MODEL_FORMAT = {
    "deepprime": "deepprime",
    "pridict2": "pridict2",
    "oped": "oped",
}


def _fetch_model_format_dataframe(req: EvaluationRequest, model_format: str) -> pd.DataFrame:
    """Resolve native model-format test data from inline records or PE-DB.

    Conversion happens in PE-DB; we request the already-converted rows from
    ``GET /api/filter`` and concatenate the matching datasheet groups.
    """
    if req.records is not None:
        return pd.DataFrame(req.records)

    missing = [
        name
        for name in ("study", "dataset", "cell_line", "pe_system")
        if getattr(req, name) is None
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide either `records`, or all of study/dataset/cell_line/pe_system "
                f"to fetch from PE-DB. Missing: {missing}"
            ),
        )

    try:
        response = requests.get(
            f"{PE_DB_URL}/api/filter",
            params={
                "study": req.study,
                "dataset": req.dataset,
                "cell_line": req.cell_line,
                "pe_system": req.pe_system,
                "format": model_format,
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch data from PE-DB: {exc}"
        ) from exc

    payload = response.json()
    frames = [
        pd.DataFrame(group["records"])
        for group in payload.get("groups", [])
        if group.get("records")
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

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
        "pe_db_url": PE_DB_URL
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


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
    """List the bundled pre-trained weight sets available for a model."""
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")
    model_class = ModelFactory._models[model_name]
    list_weights = getattr(model_class, "list_available_weights", None)
    if list_weights is None:
        return {"model": model_name, "weights": [], "count": 0}
    weights = list_weights()
    return {"model": model_name, "weights": weights, "count": len(weights)}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get predictions from a model"""
    if request.model_name not in ["deepprime", "pridict", "pridict2", "oped"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    model = ModelFactory.create_model(request.model_name, device=torch.device(DEVICE))

    return {
        "model": request.model_name,
        "predictions": [],
        "message": "Prediction endpoint - implementation pending"
    }


@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train a model on specified dataset"""
    if request.model_name not in ["deepprime", "pridict", "pridict2", "oped"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # TODO: Implement actual training logic
    return {
        "model": request.model_name,
        "dataset": f"{request.dataset_source}/{request.dataset_name}",
        "status": "training_started",
        "message": "Training endpoint - implementation pending"
    }


@app.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Evaluate a model on test data, optionally against a named pre-trained weight set.

    Test data is taken from inline `records` (standardized schema) or fetched
    from PE-DB using study/dataset/cell_line/pe_system. When `weights` is set, the
    named bundled weight set is loaded before evaluating; otherwise a freshly
    created model must already be loadable (DeepPrime/PRIDICT2 resolve defaults).
    """
    model_name = request.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    test_df = _fetch_model_format_dataframe(request, MODEL_FORMAT[model_name])
    if test_df.empty:
        raise HTTPException(status_code=404, detail="No test data resolved for evaluation.")

    try:
        model = ModelFactory.create_model(
            model_name, device=torch.device(DEVICE), **(request.model_kwargs or {})
        )

        if model_name == "oped":
            # OPED.evaluate expects tokenized inputs; encode the native OPED rows.
            prepared = model.prepare_data(test_df)
            metrics = model.evaluate(prepared, weights=request.weights)
        else:
            metrics = model.evaluate(test_df, weights=request.weights)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return {
        "model": model_name,
        "weights": request.weights,
        "n_samples": int(len(test_df)),
        "metrics": metrics,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

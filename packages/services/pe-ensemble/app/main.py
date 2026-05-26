# FastAPI endpoints for PE Ensemble service
from typing import List, Optional, Dict, Any
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.model_factory import ModelFactory
from pe_common.constants import DEVICE

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
async def evaluate_model(request: TrainingRequest):
    """Evaluate model performance"""
    if request.model_name not in ["deepprime", "pridict", "pridict2", "oped"]:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # TODO: Implement actual evaluation logic
    return {
        "model": request.model_name,
        "metrics": {},
        "message": "Evaluation endpoint - implementation pending"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

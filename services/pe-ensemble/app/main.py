# FastAPI endpoints for PE Ensemble service
from typing import List, Literal, Optional, Dict, Any, Union
import logging
import os

import pandas as pd
import requests
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from datetime import datetime, timezone

from .models.model_factory import ModelFactory
from .models import weights_registry
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
WEIGHTS_ROOT = os.getenv("WEIGHTS_ROOT", str(weights_registry.weights_root()))

# Store loaded models in memory
_loaded_models: Dict[str, Any] = {}


SplitStrategy = Literal["none", "holdout_2", "holdout_3", "cv"]


class SplitQueryParams(BaseModel):
    """PE-DB split assignment parameters (mirrors GET /api/filter)."""

    split_strategy: SplitStrategy = "none"
    train_pct: Optional[float] = None
    val_pct: Optional[float] = None
    test_pct: Optional[float] = None
    cv_folds: Optional[int] = None
    use_original_fold: bool = False
    split_random_state: int = 42
    merge: bool = False


def _default_evaluation_split() -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_2",
        train_pct=0.8,
        test_pct=0.2,
        use_original_fold=True,
    )


def _default_training_split() -> SplitQueryParams:
    return SplitQueryParams(
        split_strategy="holdout_3",
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15,
    )


class PredictionRequest(BaseModel):
    model_name: str
    sequences: List[str]
    cell_type: Optional[str] = None
    weights: Optional[str] = None


class TrainingRequest(BaseModel):
    model_name: str
    dataset_source: str
    dataset_name: str
    hyperparameters: Optional[Dict[str, Any]] = None
    split: SplitQueryParams = Field(default_factory=_default_training_split)
    study: Optional[FilterValue] = None
    dataset: Optional[FilterValue] = None
    cell_line: Optional[FilterValue] = None
    pe_system: Optional[FilterValue] = None
    edit_type: Optional[FilterValue] = None
    edit_length: Optional[FilterValue] = None
    edit_efficiency_min: Optional[float] = None
    edit_efficiency_max: Optional[float] = None
    edit_scope: Optional[FilterValue] = None
    experimental_method: Optional[FilterValue] = None
    target_context: Optional[FilterValue] = None
    scaffold_name: Optional[FilterValue] = None
    records: Optional[List[Dict[str, Any]]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


FilterScalar = Union[str, int]
FilterValue = Union[FilterScalar, List[FilterScalar]]


class EvaluationRequest(BaseModel):
    model_name: str
    # Optional bundled weight set to evaluate (see GET /models/{name}/weights).
    weights: Optional[str] = None
    # PE-DB filter params (used when `records` is not provided). Scalar or list
    # values match the PE Hub ``exportFiltered`` / ``GET /api/filter`` contract.
    study: Optional[FilterValue] = None
    dataset: Optional[FilterValue] = None
    cell_line: Optional[FilterValue] = None
    pe_system: Optional[FilterValue] = None
    edit_type: Optional[FilterValue] = None
    edit_length: Optional[FilterValue] = None
    edit_efficiency_min: Optional[float] = None
    edit_efficiency_max: Optional[float] = None
    edit_scope: Optional[FilterValue] = None
    experimental_method: Optional[FilterValue] = None
    target_context: Optional[FilterValue] = None
    scaffold_name: Optional[FilterValue] = None
    split: SplitQueryParams = Field(default_factory=_default_evaluation_split)
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


def _training_metadata_from_request(
    request: TrainingRequest,
    *,
    n_rows: int,
    train_result: Dict[str, Any],
) -> Dict[str, Any]:
    filters = {
        key: _normalize_filter_param(getattr(request, key))
        for key in (
            "study",
            "dataset",
            "cell_line",
            "pe_system",
            "edit_type",
            "edit_length",
            "edit_scope",
            "experimental_method",
            "target_context",
            "scaffold_name",
        )
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    metrics: Dict[str, Any] = {}
    if "validation_metrics" in train_result:
        metrics["validation"] = train_result["validation_metrics"]
    elif "val_pearson" in train_result:
        metrics["validation"] = {
            "pearson": train_result.get("val_pearson"),
            "spearman": train_result.get("val_spearman"),
        }
    return {
        "training": {
            "dataset_source": request.dataset_source,
            "dataset_name": request.dataset_name,
            "filters": filters,
            "split": request.split.model_dump(),
            "hyperparameters": request.hyperparameters or {},
            "model_kwargs": request.model_kwargs or {},
            "n_train_rows": n_rows,
        },
        "metrics": metrics,
        "notes": request.notes,
    }


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


def _normalize_filter_param(value: Optional[FilterValue]) -> Optional[List[FilterScalar]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    return [value]


def _build_pe_db_filter_params(
    *,
    model_format: str,
    split: Optional[SplitQueryParams] = None,
    study: Optional[FilterValue] = None,
    dataset: Optional[FilterValue] = None,
    cell_line: Optional[FilterValue] = None,
    pe_system: Optional[FilterValue] = None,
    edit_type: Optional[FilterValue] = None,
    edit_length: Optional[FilterValue] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[FilterValue] = None,
    experimental_method: Optional[FilterValue] = None,
    target_context: Optional[FilterValue] = None,
    scaffold_name: Optional[FilterValue] = None,
) -> Dict[str, Any]:
    """Build query params for PE-DB ``GET /api/filter`` (exportFiltered contract)."""
    split = split or SplitQueryParams()
    params: Dict[str, Any] = {
        "format": model_format,
        "split_strategy": split.split_strategy,
        "use_original_fold": split.use_original_fold,
        "split_random_state": split.split_random_state,
        "merge": split.merge,
    }
    if split.train_pct is not None:
        params["train_pct"] = split.train_pct
    if split.val_pct is not None:
        params["val_pct"] = split.val_pct
    if split.test_pct is not None:
        params["test_pct"] = split.test_pct
    if split.cv_folds is not None:
        params["cv_folds"] = split.cv_folds
    for name, value in (
        ("study", study),
        ("dataset", dataset),
        ("cell_line", cell_line),
        ("pe_system", pe_system),
        ("edit_type", edit_type),
        ("edit_length", edit_length),
        ("edit_scope", edit_scope),
        ("experimental_method", experimental_method),
        ("target_context", target_context),
        ("scaffold_name", scaffold_name),
    ):
        normalized = _normalize_filter_param(value)
        if normalized is not None:
            params[name] = normalized
    if edit_efficiency_min is not None:
        params["edit_efficiency_min"] = edit_efficiency_min
    if edit_efficiency_max is not None:
        params["edit_efficiency_max"] = edit_efficiency_max
    return params


def _request_pe_db_filtered(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{PE_DB_URL}/api/filter",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch data from PE-DB: {exc}"
        ) from exc
    return response.json()


def _filtered_payload_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(group["records"])
        for group in payload.get("groups", [])
        if group.get("records")
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_model_format_dataframe(
    *,
    model_format: str,
    split: SplitQueryParams,
    records: Optional[List[Dict[str, Any]]] = None,
    study: Optional[FilterValue] = None,
    dataset: Optional[FilterValue] = None,
    cell_line: Optional[FilterValue] = None,
    pe_system: Optional[FilterValue] = None,
    edit_type: Optional[FilterValue] = None,
    edit_length: Optional[FilterValue] = None,
    edit_efficiency_min: Optional[float] = None,
    edit_efficiency_max: Optional[float] = None,
    edit_scope: Optional[FilterValue] = None,
    experimental_method: Optional[FilterValue] = None,
    target_context: Optional[FilterValue] = None,
    scaffold_name: Optional[FilterValue] = None,
    evaluation: bool = False,
) -> pd.DataFrame:
    """Resolve native model-format rows from inline records or PE-DB."""
    from pe_common.splits import select_evaluation_partition

    if records is not None:
        df = pd.DataFrame(records)
        if evaluation and split.split_strategy != "none":
            return select_evaluation_partition(df, require_test=True)
        return df

    params = _build_pe_db_filter_params(
        model_format=model_format,
        split=split,
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
    payload = _request_pe_db_filtered(params)
    df = _filtered_payload_to_dataframe(payload)
    if evaluation and split.split_strategy != "none":
        return select_evaluation_partition(df, require_test=True)
    return df


def _fetch_evaluation_dataframe(req: EvaluationRequest, model_format: str) -> pd.DataFrame:
    return _fetch_model_format_dataframe(
        model_format=model_format,
        split=req.split,
        records=req.records,
        study=req.study,
        dataset=req.dataset,
        cell_line=req.cell_line,
        pe_system=req.pe_system,
        edit_type=req.edit_type,
        edit_length=req.edit_length,
        edit_efficiency_min=req.edit_efficiency_min,
        edit_efficiency_max=req.edit_efficiency_max,
        edit_scope=req.edit_scope,
        experimental_method=req.experimental_method,
        target_context=req.target_context,
        scaffold_name=req.scaffold_name,
        evaluation=True,
    )

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
    return _request_pe_db_filtered(params)


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
            device=torch.device(DEVICE),
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


@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train a model on PE-DB data with centralized split assignments."""
    model_name = request.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        train_df = _fetch_model_format_dataframe(
            model_format=MODEL_FORMAT[model_name],
            split=request.split,
            records=request.records,
            study=request.study,
            dataset=request.dataset,
            cell_line=request.cell_line,
            pe_system=request.pe_system,
            edit_type=request.edit_type,
            edit_length=request.edit_length,
            edit_efficiency_min=request.edit_efficiency_min,
            edit_efficiency_max=request.edit_efficiency_max,
            edit_scope=request.edit_scope,
            experimental_method=request.experimental_method,
            target_context=request.target_context,
            scaffold_name=request.scaffold_name,
            evaluation=False,
        )
        if train_df.empty:
            raise HTTPException(status_code=404, detail="No training data resolved.")

        from pe_common.splits import exclude_test_partition

        train_df = exclude_test_partition(train_df)
        if train_df.empty:
            raise HTTPException(status_code=422, detail="No non-test rows available for training.")

        model = ModelFactory.create_model(
            model_name, device=torch.device(DEVICE), **(request.model_kwargs or {})
        )
        if model_name == "oped":
            train_df = model.prepare_data(train_df)

        result = model.train(train_df, hyperparameters=request.hyperparameters)

        metadata = _training_metadata_from_request(
            request,
            n_rows=int(len(train_df)),
            train_result=result,
        )
        weights_id = weights_registry.register_trained_model(
            model_name,
            model,
            metadata=metadata,
            notes=request.notes,
        )
        entry = weights_registry.get_manifest(model_name, weights_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {
        "model": model_name,
        "status": "success",
        "split_strategy": request.split.split_strategy,
        "n_rows": int(len(train_df)),
        "weights_id": weights_id,
        "weights_label": entry.get("label"),
        "result": result,
    }


@app.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Evaluate a model on test data, optionally against a named pre-trained weight set.

    Test data is taken from inline `records` (model-native schema) or fetched
    from PE-DB via the same multi-value filters as ``GET /data/filter`` /
    PE Hub ``exportFiltered``. When PE-DB split assignments are present, only
    rows with ``split=test`` are evaluated.
    """
    model_name = request.model_name.strip().lower()
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    test_df = _fetch_evaluation_dataframe(request, MODEL_FORMAT[model_name])
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

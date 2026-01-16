"""PE Database API - Main FastAPI Application

This API serves prime editing efficiency data from various sources
in different formats for model training and evaluation.
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, Optional, List, Dict
import pandas as pd
from pathlib import Path
import logging

from .data_prep.converter import DataConverter
from .data_prep.loaders import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PE Database API",
    description="API for serving prime editing efficiency data in various formats",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data services
converter = DataConverter()
loader = DataLoader()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "PE Database API",
        "version": "0.1.0",
        "description": "API for serving prime editing efficiency data",
        "endpoints": {
            "data": "/api/data - Get data in requested format",
            "datasets": "/api/datasets - List available datasets",
            "convert": "/api/convert - Convert raw data to standardized format",
            "export": "/api/export/deepprime - Export all DeepPrime data"
        }
    }


@app.get("/api/data")
async def get_data(
    cell_line: str = Query(..., description="Cell line (e.g., HEK293T, A549, DLD1)"),
    pe_system: str = Query(..., description="PE system (e.g., PE2, PE2max, PE4max)"),
    source_model: str = Query(..., description="Source model (dp, dp_ft, pd1, pd2, etc.)"),
    target_format: Literal["std", "oped", "deepprime", "pridict", "pridict2"] = Query(
        "std", 
        description="Target format for the data"
    ),
    limit: Optional[int] = Query(None, description="Limit number of records returned")
):
    """
    Get prime editing data in requested format
    
    Returns data from the specified source in the target format.
    If target format doesn't exist, attempts to load from standardized format.
    
    Parameters:
    - **cell_line**: Cell line name (e.g., HEK293T, A549, DLD1)
    - **pe_system**: Prime editing system (e.g., PE2, PE2max, PE4max)
    - **source_model**: Source model identifier (dp, dp_ft, pd1, pd2, etc.)
    - **target_format**: Format to return data in (std, oped, deepprime, pridict, pridict2)
    - **limit**: Optional limit on number of records returned
    
    Returns:
    - JSON object with data and metadata
    """
    try:
        # Load data in target format
        data = loader.load_data(
            cell_line=cell_line,
            pe_system=pe_system,
            source_model=source_model,
            target_format=target_format
        )
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            data = data.head(limit)
        
        return {
            "status": "success",
            "metadata": {
                "cell_line": cell_line,
                "pe_system": pe_system,
                "source_model": source_model,
                "format": target_format,
                "total_records": len(data),
                "columns": list(data.columns)
            },
            "data": data.to_dict(orient="records")
        }
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/api/datasets")
async def list_datasets():
    """
    List all available datasets in the database
    
    Returns:
    - List of available datasets with metadata
    """
    try:
        datasets = loader.list_available_datasets()
        
        return {
            "status": "success",
            "count": len(datasets),
            "datasets": datasets.to_dict(orient="records")
        }
    
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.post("/api/convert")
async def convert_data(
    source: Literal["deepprime", "pridict", "pridict2"],
    cell_line: str = Query(..., description="Cell line name"),
    pe_system: str = Query(..., description="PE system"),
    model_variant: Optional[str] = Query(None, description="Model variant (e.g., dp_ft)")
):
    """
    Convert raw data to standardized format
    
    Parameters:
    - **source**: Data source format (deepprime, pridict, pridict2)
    - **cell_line**: Cell line name
    - **pe_system**: PE system
    - **model_variant**: Optional model variant
    
    Returns:
    - Conversion result with record count
    """
    try:
        result = converter.convert_to_standardized(
            source=source,
            cell_line=cell_line,
            pe_system=pe_system,
            model_variant=model_variant
        )
        
        return {
            "status": "success",
            "message": f"Successfully converted {source} data to standardized format",
            "records_converted": len(result),
            "output_columns": list(result.columns)
        }
    
    except FileNotFoundError as e:
        logger.error(f"Source file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error converting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting data: {str(e)}")


@app.post("/api/export/deepprime")
async def export_deepprime_all():
    """
    Export all sheets from the original DeepPrime Excel file to CSV format
    
    This endpoint processes the original DeepPrime data file and exports
    all sheets to individual CSV files in the deepprime directory.
    
    Returns:
    - Export result with number of files created
    """
    try:
        converter.export_deepprime_all()
        
        return {
            "status": "success",
            "message": "Successfully exported all DeepPrime data to CSV files",
        }
    
    except FileNotFoundError as e:
        logger.error(f"DeepPrime source file not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting DeepPrime data: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

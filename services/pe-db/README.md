# PE Database Service

A FastAPI-based service that serves and converts Prime Editing datasets into standardized or model-specific formats.

## Features

- Serves standardized PE datasets from multiple sources (DeepPrime, PRIDICT, MinSEPIE, OPED)
- Converts raw datasets into standardized/model-specific formats
- RESTful API for data retrieval
- Flexible data filtering and formatting

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /datasets` - List available datasets
- `GET /datasets/{dataset_id}` - Get specific dataset
- `POST /query` - Query datasets with filters

## Running Locally

```bash
cd services/pe-db
uvicorn pe_db.main:app --reload --port 8000
```

The service will be available at http://localhost:8000

## Development

```bash
# Install dependencies
pip install -e .

# Run development server
uvicorn pe_db.main:app --reload --port 8000
```

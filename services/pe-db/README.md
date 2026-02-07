# PE Database Service

A FastAPI-based service that serves and converts Prime Editing datasets into standardized or model-specific formats.

## Features

- Serves standardized PE datasets from multiple sources (DeepPrime, PRIDICT, MinSEPIE, OPED)
- Converts raw datasets into standardized/model-specific formats
- RESTful API for data retrieval
- Flexible data filtering and formatting
- Docker-based deployment

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /datasets` - List available datasets
- `GET /datasets/{dataset_id}` - Get specific dataset
- `POST /query` - Query datasets with filters

## Running Locally

```bash
cd services/pe-db
docker-compose up
```

The service will be available at http://localhost:8000

## Running as Part of Full Stack

```bash
cd ../..
docker-compose -f docker-compose.full.yml up pe-db
```

## Development

```bash
# Install dependencies
pip install -e .

# Run development server
uvicorn pe_db.main:app --reload --port 8000
```

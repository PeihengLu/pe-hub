# PE Ensemble Service

A FastAPI-based web service for evaluating, training, and creating ensembles of Prime Editing efficiency prediction models.

## Features

- Access to state-of-the-art PE prediction models (DeepPrime, PRIDICT, PRIDICT2, OPED)
- Conventional ML baselines (for example, XGBoost) alongside vendor models
- Model training and evaluation endpoints
- Ensemble prediction capabilities
- Integration with PE Database service for data retrieval

## Models Supported

- **DeepPrime** - Deep learning model for PE efficiency prediction
- **PRIDICT** - Position-specific scoring matrix model
- **PRIDICT2** - Improved version of PRIDICT
- **OPED** - Optimized Prime Editor prediction model

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /models` - List available models
- `POST /train` - Train a model
- `POST /predict` - Get predictions from a model
- `POST /evaluate` - Evaluate model performance
- `POST /ensemble` - Get ensemble predictions

## Running Locally

```bash
cd services/pe-ensemble
docker-compose up
```

This compose file starts both `pe-ensemble` and its `pe-db` dependency.
The service will be available at http://localhost:8001 by default.

## Running as Part of Full Stack

```bash
cd ../..
docker-compose -f docker-compose.full.yml up
```

## Development

```bash
# Install dependencies
pip install -e .

# Run development server
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001
```

## Dependencies

This service depends on:
- PE Database service for data retrieval
- Vendor model repositories under `vendor/models/` (each subdirectory is a separate model codebase)

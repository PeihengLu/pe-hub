# PE-DB Quick Reference Card

## 🚀 Quick Start

```bash
# Option 1: Docker (Full Stack)
docker-compose -f docker-compose.full.yml up -d

# Option 2: Local Development
make install && bash setup.sh
```

## 📡 Service URLs

| Service | URL | Docs |
|---------|-----|------|
| PE Database | http://localhost:8000 | /docs |
| PE Ensemble | http://localhost:8001 | /docs |

## 🔧 Common Commands

### Makefile Commands
```bash
make install        # Install dependencies
make docker-up      # Start all services
make docker-down    # Stop all services
make docker-logs    # View logs
make docker-build   # Build images
make jupyter        # Start Jupyter locally
make test           # Run tests
make format         # Format code
make lint           # Lint code
make clean          # Clean generated files
```

### Docker Commands
```bash
# Full stack
docker-compose -f docker-compose.full.yml up -d
docker-compose -f docker-compose.full.yml down
docker-compose -f docker-compose.full.yml logs -f

# Individual services
cd services/pe-db && docker-compose up -d
cd services/pe-ensemble && docker-compose up -d

# Debug
docker exec -it pe-db bash
docker logs pe-ensemble -f
```

### Local Development
```bash
# PE Database
cd services/pe-db
uvicorn pe_db.main:app --reload --port 8000

# PE Ensemble
cd services/pe-ensemble
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001

# Jupyter
jupyter lab
```

## 📚 API Examples

### PE Database API

```bash
# Health check
curl http://localhost:8000/health

# List datasets
curl http://localhost:8000/datasets

# Get specific dataset
curl http://localhost:8000/datasets/deepprime/dp-hek293t-pe2

# Get dataset info
curl http://localhost:8000/datasets/deepprime/dp-hek293t-pe2/info
```

### PE Ensemble API

```bash
# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/models

# Predict
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepprime", "sequences": ["ATCG..."]}'

# Train
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "deepprime", "dataset_source": "deepprime", "dataset_name": "dp-hek293t-pe2"}'
```

## 🐍 Python Usage

### In Jupyter Notebooks

```python
# Import utilities
from src import data, features, sequence_utils
from pe_common import utils

# Access PE Database
import requests
import pandas as pd

response = requests.get("http://localhost:8000/datasets")
datasets = response.json()

# Load local dataset
df = pd.read_csv("../datasets/standardized/deepprime/dp-hek293t-pe2.csv")

# Use models
import torch
from src.train_models import train_model
from src.run_models import predict
```

### In Services

```python
# services/pe-db/pe-db/main.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/my-endpoint")
async def my_endpoint():
    return {"status": "ok"}

# services/pe-ensemble/pe-ensemble/main.py
import requests
PE_DB_URL = "http://pe-db:8000"
response = requests.get(f"{PE_DB_URL}/datasets")
```

## 📁 Project Structure

```
pe-db/
├── services/
│   ├── pe-db/           # Database API service
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── pe-db/main.py
│   │   └── data_prep/
│   └── pe-ensemble/     # Model service
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── pe-ensemble/main.py
├── packages/
│   └── pe-common/       # Shared utilities
├── datasets/
│   ├── raw/            # Original data
│   └── standardized/   # Processed CSV
├── vendor/models/      # Model weights
├── src/               # Core utilities
├── analysis/          # Jupyter notebooks
└── docker-compose.full.yml
```

## 🔄 Development Workflow

1. **Make Changes** → Edit code in services/notebooks
2. **Format** → `make format`
3. **Test** → `make test`
4. **Restart** → `docker-compose restart <service>`
5. **View Logs** → `make docker-logs`
6. **Commit** → Git commit changes

## 🐛 Troubleshooting

### Services won't start
```bash
docker-compose -f docker-compose.full.yml down -v
docker-compose -f docker-compose.full.yml build --no-cache
docker-compose -f docker-compose.full.yml up
```

### Import errors
```bash
pip install -e . && pip install -e packages/pe-common
```

### Data not found
```bash
bash setup.sh
```

### Port already in use
```bash
# Change ports in docker-compose.full.yml
# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

## 📖 Documentation

- **README.md** - Project overview
- **DEVELOPMENT.md** - Developer guide
- **SETUP_COMPLETE.md** - Setup summary
- **/docs** - API documentation (each service)

## 🎯 Supported Models

| Model | Type | Description |
|-------|------|-------------|
| DeepPrime | Neural Network | Deep learning PE predictor |
| PRIDICT | PSSM | Position-specific scoring |
| PRIDICT2 | PSSM | Improved PRIDICT |
| OPED | Neural Network | Optimized PE model |

## 🗂️ Dataset Sources

- **DeepPrime** - `datasets/standardized/deepprime/`
- **PRIDICT** - `datasets/raw/pridict1-org/`
- **PRIDICT2** - `datasets/raw/pridict2-org/`
- **MinSEPIE** - `datasets/raw/minsepie-org/`

## ⚙️ Environment Variables

```bash
# .env file
DATA_PATH=./datasets/standardized
MODEL_PATH=./vendor/models
PE_DB_URL=http://localhost:8000
PE_ENSEMBLE_URL=http://localhost:8001
CUDA_VISIBLE_DEVICES=0
JUPYTER_TOKEN=
```

## 🧪 Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov=services

# Specific file
pytest tests/test_data.py -v
```

## 📦 Package Management

```bash
# Install main project
pip install -e .

# Install with dev dependencies
pip install -e .[dev]

# Install pe-common
pip install -e packages/pe-common

# Update requirements
pip freeze > requirements.txt
```

---

**Need Help?** Check DEVELOPMENT.md for detailed guide

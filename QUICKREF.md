# PE-DB Quick Reference

## Quick start

```bash
./scripts/start-all.sh --install   # first time
./scripts/start-all.sh
```

Or set up Python manually:

```bash
make setup && source venv/bin/activate
make install
```

## Service URLs

| Service | URL | API docs |
|---------|-----|----------|
| PE Hub (UI) | http://localhost:5173 | — |
| PE Database | http://localhost:8000 | /docs |
| PE Ensemble | http://localhost:8001 | /docs |

## Start services

```bash
# All three (recommended)
./scripts/start-all.sh

# PE Database only
./scripts/start-pe-db-backend.sh --install

# PE Database headless (no HTTP)
cd services/pe-db && pe-db init

# Individual backends
cd services/pe-db && uvicorn app.main:app --reload --port 8000
cd services/pe-ensemble && PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8001
cd pe-hub && npm run dev
```

## Common commands

```bash
make install        # Install editable packages + dev deps
make test           # Run pytest
make format         # black
make lint           # flake8
make clean          # Remove caches and build artifacts
make data-prep      # Legacy dataset prep (scripts/setup.sh)
```

## API examples

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health

# Catalog
curl http://localhost:8000/api/studies
curl http://localhost:8000/api/filter?study=deepprime

# Model-format export with split
curl "http://localhost:8000/api/filter?format=deepprime&study=pridict1&dataset=library2&cell_line=HEK293T&pe_system=PE2max&split_strategy=holdout_2&train_pct=0.8&test_pct=0.2"

# List models and devices
curl http://localhost:8001/models
curl http://localhost:8001/devices
```

## Environment variables

| Variable | Service | Purpose |
|----------|---------|---------|
| `PE_DB_URL` | Ensemble | PE Database base URL when `PE_DB_MODE=http` (default `http://localhost:8000`) |
| `PE_DB_MODE` | Ensemble | `http` (default) or `library` for in-process PE-DB access |
| `WEIGHTS_ROOT` | Ensemble | Override weights directory |
| `TRAINING_JOBS_ROOT` | Ensemble | Override `jobs/` location |
| `DATABASE_URL` | Database | Override SQLite catalog path |
| `PE_DB_FORCE_EXPORT` | Database | Re-export on startup |
| `PE_DB_FORCE_STANDARDIZE` | Database | Re-standardize on startup |
| `VITE_PE_DB_URL` | PE Hub | Database API URL |
| `VITE_ENSEMBLE_API_URL` | PE Hub | Ensemble API URL |

## Troubleshooting

### Import errors

```bash
pip install -e packages/pe-common
pip install -e services/pe-ensemble
cd services/pe-db && pip install -r requirements.txt
```

### Data not found

Start PE Database once so export/standardize runs, or force re-export:

```bash
./scripts/start-all.sh --force-reexport
```

### Port already in use

```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

## Documentation

| Path | Contents |
|------|----------|
| `README.md` | Project overview |
| `services/pe-db/README.md` | Catalog schema and database init |
| `services/pe-ensemble/README.md` | Models, training, weights |
| `services/pe-ensemble/jobs/README.md` | Training job filesystem layout |
| `packages/pe-common/README.md` | Shared utilities |
| `pe-hub/README.md` | Frontend setup |

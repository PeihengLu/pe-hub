# PE-DB Quick Reference Card

## Quick Start

```bash
make install
bash setup.sh
```

## Service URLs

| Service | URL | Docs |
|---------|-----|------|
| PE Hub (UI) | http://localhost:5173 | — |
| PE Database | http://localhost:8000 | /docs |
| PE Ensemble | http://localhost:8001 | /docs |

## Start all services

```bash
./start-all.sh --install   # first time
./start-all.sh
```

## Common Commands

### Makefile Commands
```bash
make install        # Install dependencies
make test           # Run tests
make format         # Format code
make lint           # Lint code
make clean          # Clean generated files
make jupyter        # Start Jupyter locally
```

### Local Development
```bash
# PE Database
cd services/pe-db
uvicorn pe_db.main:app --reload --port 8000

# PE Ensemble
cd services/pe-ensemble
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001
```

## API Examples

```bash
# PE Database health check
curl http://localhost:8000/health

# PE Ensemble health check
curl http://localhost:8001/health
```

## Troubleshooting

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
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
```

## Documentation

- `README.md` - Project overview
- `DEVELOPMENT.md` - Developer guide
- `SETUP_COMPLETE.md` - Setup summary

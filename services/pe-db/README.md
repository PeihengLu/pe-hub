# PE Database Service

FastAPI service aligned with `diagrams/illustration/database_er.mmd`.

## What lives in the catalog database

| Table | How it is maintained |
|-------|----------------------|
| `study` | **Seeded** — hand-coded in `app/catalog/studies.py` |
| `dataset` | **Seeded** — hand-coded in `app/catalog/studies.py` |
| `scaffold` | **Seeded** — hand-coded in `app/catalog/scaffolds.py` |
| `datasheet` | **Semi-automatic** — indexed from `datasets/exported/**/*.csv` after export |

Edit-level rows are **not** in SQL. They live in `datasets/exported/` (raw CSV) and `datasets/standardized/` (parquet) and are loaded with Pandas behind the API.

SQLite path: `datasets/catalog/pe_database.db` (override with `DATABASE_URL`).

## Database initialization (three steps)

`initialize_database()` runs on API startup:

| Step | Function | What it does |
|------|----------|--------------|
| 1. **Seed** | `init_catalog()` | Creates SQL tables; inserts/updates Study, Dataset, Scaffold from Python registries |
| 2. **Export** | `export_original_data()` | Writes `datasets/exported/` from raw Excel/CSV; indexes **Datasheet** rows |
| 3. **Standardize** | `standardize_exported_data()` | Writes `datasets/standardized/` parquet from exported CSVs |

### What “seed” means

**Seed** = populate fixed lookup tables from code you maintain by hand.

It is the same idea as `db/seeds/` in Rails or `INSERT` scripts for reference data: the database gets the canonical list of studies (DeepPrime, PRIDICT, …), datasets (`deepprime-clinvar`, `library1`, …), and pegRNA scaffolds (conventional, optimized, …) before any file scanning happens.

Seed does **not**:

- Read edit data from disk
- Export or standardize measurements
- Fill the `Datasheet` table (that happens during **export**, when CSV paths are registered)

## Manual usage

### CLI (no HTTP server)

```bash
cd services/pe-db
pip install -e ../../packages/pe-common --no-deps
pip install -e .

pe-db init                              # seed + export + standardize
pe-db init --force-export               # re-export raw study files
pe-db export --study deepprime
pe-db standardize --force
pe-db filter --format deepprime --dataset library2 \
  --cell-line HEK293T --pe-system PE2max --split-strategy holdout_3 \
  --out /tmp/deepprime_train.parquet
pe-db studies
pe-db datasets --study deepprime
pe-db datasheets
pe-db scaffolds
pe-db statistics
pe-db formats
pe-db plugins reload
```

### Python

```python
from app.catalog.initialize import initialize_database
from app.converter import DataConverter
from pe_db.library import filter_from_params, run_init

run_init()  # seed + export + standardize
# or
initialize_database()  # same pipeline via app.catalog
```

```bash
curl -X POST 'http://localhost:8000/api/export'
curl -X POST 'http://localhost:8000/api/convert?study=deepprime&dataset=deepprime-clinvar&cell_line=hek293t&pe_system=pe2'
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/studies` | List studies |
| GET | `/api/datasets` | List datasets (optional `?study=` filter) |
| GET | `/api/datasheets` | List datasheet catalog entries |
| GET | `/api/scaffolds` | List pegRNA scaffolds |
| GET | `/api/data` | Load standardized edit records for one datasheet |
| GET | `/api/filter` | Filter catalog and/or export model-format data with train/val/test splits |
| GET | `/api/statistics` | Aggregate statistics (edit type, length, delivery method, …) |
| POST | `/api/export` | Export (+ optional standardize) |
| POST | `/api/convert` | Standardize one sheet |
| GET | `/health` | Health check |

### Filter and export

`GET /api/filter` serves two modes:

- **Without `format`** — returns matching datasheet metadata (catalog browse).
- **With `format`** (`std`, `deepprime`, `pridict`, `pridict2`, `oped`) — converts
  standardizable datasets from parquet into the requested model schema. Requires
  `split_strategy` (`none`, `holdout_2`, `holdout_3`, `cv`). Supports multi-value
  filters (`cell_line=HEK293T&cell_line=A549`), efficiency bounds, scaffold name,
  and `merge=true` to combine datasheets before split assignment.

PE Ensemble and PE Hub call this endpoint (Ensemble proxies it at `GET /data/filter`).

## Run locally

### pip (default)

```bash
cd services/pe-db
pip install -r requirements.txt
pip install -e ../../packages/pe-common --no-deps
pip install -e .   # installs ``pe-db`` CLI and ``pe_db`` library package
uvicorn app.main:app --reload --port 8000
```

Or from the repo root: `./scripts/start-pe-db-backend.sh --install`

### conda (recommended when using DeepSpCas9 scoring)

Conda-forge provides TensorFlow builds for macOS, Linux, and Windows without
platform-specific pip wheels. The startup scripts use whichever Python is active
(`conda activate …` or a venv).

```bash
conda create -n pe-db python=3.11 -y
conda activate pe-db
conda install -c conda-forge tensorflow -y   # optional; for MinSePIE spcas9 backfill

cd services/pe-db
pip install -r requirements.txt
pip install -e ../../packages/pe-common --no-deps
uvicorn app.main:app --reload --port 8000
```

TensorFlow is **optional**. Standardization runs without it; rows that need a
computed `spcas9_score` stay `NaN` and a warning is logged. With conda TensorFlow
installed, MinSePIE standardization fills scores via `fill_missing_spcas9_scores`.

Re-standardize after enabling scoring:

```bash
PE_DB_FORCE_STANDARDIZE=1 ./scripts/start-pe-db-backend.sh
# or: curl -X POST 'http://localhost:8000/api/export?study=minsepie&force_standardize=true'
```

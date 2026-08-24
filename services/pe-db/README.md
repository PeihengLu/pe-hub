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

## Architecture

```text
pedb (pe-db)  ──┐
                ├──► pe_db.library  (← app/library.py)
FastAPI         ──┘
peen (pe-ensemble) CLI ──► pe_db.library (in-process; no PE-DB HTTP server)
pe-ensemble web ──HTTP──► FastAPI
```

Installable packages: `pe_db` (console scripts `pedb` / `pe-db` + `pe_db.library`) and the FastAPI app under `app/`. Both the CLI and HTTP handlers call the same headless library — no localhost hop for CLI use.

## Manual usage

### CLI (no HTTP server)

```bash
# Activate conda/venv first (not system Python)
pip install -e packages/pe-common
pip install -e services/pe-db
```

Or from the repo root (same requirement): `./scripts/install-clis.sh` (installs **pedb** / **peen** and bash/zsh tab completion, then prints usage). Reload completion with `conda deactivate && conda activate pedb`, then `pedb <TAB>`. Skip completion with `SKIP_CLI_COMPLETION=1 ./scripts/install-clis.sh`.

Short alias: **`pedb`** (also installed as `pe-db`).

| Command | Purpose |
|---------|---------|
| `pedb init` | Seed catalog, export raw data, standardize |
| `pedb seed` | Seed Study / Dataset / Scaffold tables only |
| `pedb export` | Export raw study files (+ optional standardize) |
| `pedb standardize` | Exported CSV → parquet |
| `pedb convert` | Standardize one datasheet (`--study` `--dataset` `--cell-line` `--pe-system`) |
| `pedb filter` | Same contract as `GET /api/filter` (catalog and/or model-format export) |
| `pedb studies` | List catalog studies |
| `pedb datasets` | List datasets (optional `--study`) |
| `pedb datasheets` | List datasheets (optional `--study` / `--dataset`) |
| `pedb scaffolds` | List pegRNA scaffolds |
| `pedb statistics` | Descriptive stats over edit rows |
| `pedb formats` | List supported filter output formats |
| `pedb plugins reload` | Reload active plugin converters |

Examples:

```bash
pedb init                              # seed + export + standardize
pedb init --force-export               # re-export raw study files
pedb seed
pedb export --study deepprime
pedb standardize --force
pedb convert --study deepprime --dataset deepprime-clinvar \
  --cell-line hek293t --pe-system pe2
pedb filter --format deepprime --dataset library2 \
  --cell-line HEK293T --pe-system PE2max --split-strategy holdout_3 \
  --out /tmp/deepprime_train.parquet
pedb studies
pedb datasets --study deepprime
pedb datasheets --study deepprime --dataset library2
pedb scaffolds
pedb statistics --edit-type sub
pedb formats
pedb plugins reload
```

`pedb filter --out` accepts `.json` (full payload), `.csv`, or `.parquet` (merged rows).

### Python library

Prefer the installable package (works from any cwd; also used by the `peen` CLI):

```python
from pe_db.library import filter_from_params, list_studies, run_init

run_init()  # seed + export + standardize
print(list_studies())
payload = filter_from_params(
    {
        "format": "deepprime",
        "dataset": ["library2"],
        "cell_line": ["HEK293T"],
        "pe_system": ["PE2max"],
        "split_strategy": "holdout_3",
        "train_pct": 0.7,
        "val_pct": 0.15,
        "test_pct": 0.15,
    }
)
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

CLI mirrors the same flags (`pedb filter --merge --use-original-fold …`).

#### Splits, author folds, and merge

| Flag | Effect |
|------|--------|
| `--split-strategy` | `none` / `holdout_2` / `holdout_3` / `cv` |
| `--use-original-fold` | Prefer author `original_fold` where present (`-1` = permanent test by default via `--original-fold-test-value`) |
| `--merge` | Concatenate matching datasheets, then reassign groups by shared protospacer before splitting |

**Merged DeepPrime ClinVar + PRIDICT library1:** with `--merge --use-original-fold`, PE-DB runs `propagate_original_fold_by_target_uid` after concatenation so library1 rows that share a protospacer/`target_uid` with ClinVar inherit DeepPrime’s `original_fold`. Non-overlapping library1 loci keep random CV / holdout groups. This is what the PRIDICT 2.0 reproduction base uses (see below).

Example (merged export for training clients):

```bash
pedb filter --format pridict2 \
  --study pridict1 --dataset library1 \
  --study deepprime --dataset deepprime-clinvar \
  --cell-line hek293t --pe-system pe2 \
  --merge --use-original-fold --original-fold-test-value -1 \
  --split-strategy cv --cv-folds 5 --test-pct 0.15 \
  --out /tmp/l1_plus_clinvar.parquet
```

PE Ensemble’s **web service** and PE Hub call this endpoint over HTTP (Ensemble proxies it at `GET /data/filter`). The **`peen` CLI** uses the same filter logic in-process via `pe_db.library` (no PE-DB server required).

### Experiment scripts that use PE-DB

Dataset HPO recipes and the PRIDICT 2.0 transfer + ensemble reproduction call `peen`, which loads data through `pe_db.library` (same path as `pedb filter`):

- [`scripts/experiments/`](../../scripts/experiments/README.md) — per-dataset HPO recipes
- [`scripts/experiments/pridict2-reproduction/`](../../scripts/experiments/pridict2-reproduction/README.md) — full base → fine-tune → mean-ensemble pipeline (library1, L1+ClinVar merge, library-diverse)

## Run locally

### pip (default)

```bash
pip install -e packages/pe-common
pip install -e services/pe-db   # installs ``pedb`` / ``pe-db`` CLI and ``pe_db`` library package
cd services/pe-db
uvicorn app.main:app --reload --port 8000
```

Or from the repo root: `./scripts/start-pe-db-backend.sh --install`, or `./start-all.sh --install` (also installs pe-ensemble).

### conda (recommended when using DeepSpCas9 scoring)

Conda-forge provides TensorFlow builds for macOS, Linux, and Windows without
platform-specific pip wheels. The startup scripts use whichever Python is active
(`conda activate …` or a venv).

```bash
conda create -n pe-db python=3.11 -y
conda activate pe-db
conda install -c conda-forge tensorflow -y   # optional; for MinSePIE spcas9 backfill

pip install -e packages/pe-common
pip install -e services/pe-db
cd services/pe-db
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

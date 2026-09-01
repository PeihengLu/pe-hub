# PE-DB: Prime Editing Database and Model Ensemble

A platform for prime editing efficiency data management, model evaluation, and training.

## Components

| Component                                        | Port | Role                                                                    |
| ------------------------------------------------ | ---- | ----------------------------------------------------------------------- |
| **PE Database** (`services/pe-db`)       | 8000 | Catalog metadata (SQLite) + standardized edit records (parquet/CSV)     |
| **PE Ensemble** (`services/pe-ensemble`) | 8001 | Model wrappers, training jobs, evaluation, weight registry              |
| **PE Hub** (`pe-hub`)                    | 5173 | Unified React UI for catalog browsing, export, training, and evaluation |

Shared Python utilities live in `packages/pe-common`.

## Installation

PE-Hub requires **Python 3.11** (CLI and web portal). Python 3.13+ is unsupported —
OptiPrime's `rs3` dependency cannot install on newer interpreters.

**Prerequisites:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
Mambaforge (recommended), or `python3.11` on your PATH for a local venv. Do not use
Apple Command Line Tools / system Python.

### Quick start

From the repository root:

**CLI only** (`pedb` / `peen` — catalog, training, evaluation without HTTP servers):

```bash
./scripts/setup-python-env.sh --install
conda activate pe-hub
```

**Web portal** (PE Hub UI + PE Database + PE Ensemble APIs):

```bash
./scripts/setup-python-env.sh --install
conda activate pe-hub
./scripts/start-all.sh --install   # npm deps for the frontend
./scripts/start-all.sh
```

- PE Hub: http://localhost:5173
- PE Database API docs: http://localhost:8000/docs
- PE Ensemble API docs: http://localhost:8001/docs

`setup-python-env.sh --install` creates a Python 3.11 conda environment (`pe-hub` by
default), installs ViennaRNA and Node.js via conda, then runs `install-clis.sh`
(vendor submodules, `pedb`, `peen`, OptiPrime/JAX deps, tab completion).

In new terminal sessions, activate the env before using the CLIs or starting services:

```bash
conda activate pe-hub
```

### Setup scripts

| Script | What it does |
|--------|----------------|
| `./scripts/setup-python-env.sh --install` | Create/update Python 3.11 env **and** install project packages |
| `./scripts/install-clis.sh` | Install packages only (env must already be active) |
| `./scripts/start-all.sh --install` | Run `install-clis.sh` + `npm install`, then start all services |

Environment recipe: `environment.yml`. Pip version pins: `requirements/constraints.txt`.

### Options

```bash
# Custom conda env name (e.g. pedb)
./scripts/setup-python-env.sh --name pedb --install

# Local venv instead of conda (requires python3.11 on PATH)
./scripts/setup-python-env.sh --venv --install
source venv/bin/activate

# Skip OptiPrime / JAX stack (DeepPrime, OPED, PRIDICT2 only)
SKIP_OPTIPRIME=1 ./scripts/install-clis.sh

# Reload tab completion after install
conda deactivate && conda activate pe-hub
```

### Verify

```bash
pedb studies
peen models
```

Details: [`services/pe-db/README.md`](services/pe-db/README.md), [`services/pe-ensemble/README.md`](services/pe-ensemble/README.md).

### Advanced / manual setup

Run backends individually (after `./scripts/install-clis.sh`):

```bash
# PE Database
cd services/pe-db
uvicorn app.main:app --reload --port 8000

# PE Ensemble
cd services/pe-ensemble
PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8001

# PE Hub
cd pe-hub && npm install && npm run dev
```

Legacy dataset prep (optional): `bash scripts/setup.sh`

Vendor model source code is under `vendor/models/` (git submodules). Pretrained
weights are versioned in `services/pe-ensemble/weights/` — see
`services/pe-ensemble/weights/README.md`.

## Project structure

```
pe-db/
├── packages/pe-common/       # Shared constants, splits, devices, training helpers
├── pe-hub/                   # Unified web UI
├── services/
│   ├── pe-db/                # FastAPI catalog + data service
│   │   └── app/
│   │       ├── catalog/      # Study/dataset/scaffold registries (seeded)
│   │       ├── converter.py  # Export + standardize pipeline
│   │       ├── db/           # SQLAlchemy catalog repository
│   │       └── utils/        # Standardization and format conversion
│   └── pe-ensemble/          # FastAPI model service
│       ├── app/
│       │   ├── models/       # DeepPrime, OPED, PRIDICT2 wrappers
│       │   └── training/     # Job queue, device scheduler, runner
│       ├── jobs/             # Filesystem-backed training job state
│       └── weights/          # Registered pretrained + trained checkpoints
├── datasets/
│   ├── raw/                  # Original study files (Excel, CSV, …)
│   ├── exported/             # Normalized CSV per datasheet (generated)
│   ├── standardized/         # Parquet in shared schema (generated)
│   └── catalog/              # SQLite catalog DB (generated)
├── vendor/models/            # Third-party model code (submodules)
├── scripts/                  # start-all, start-pe-db-backend, smoke tests, setup
└── Makefile                  # install, test, lint, format
```

## Data pipeline

On PE Database startup, `initialize_database()` runs three steps:

1. **Seed** — create SQL tables; insert studies, datasets, and scaffolds from Python registries (`app/catalog/`)
2. **Export** — write `datasets/exported/` from raw files; register **Datasheet** rows in the catalog
3. **Standardize** — write `datasets/standardized/` parquet from exported CSVs

Edit-level measurements are **not** stored in SQL. They are loaded with Pandas from parquet/CSV behind the API. Catalog tables (`study`, `dataset`, `scaffold`, `datasheet`) are described in `services/pe-db/README.md` and `diagrams/illustration/database_er.mmd`.

Supported studies include DeepPrime, PRIDICT1, PRIDICT2, MinsePIE, and DeepPE (see `app/catalog/studies.py`).

### Output formats

| Format                     | Use                                                     |
| -------------------------- | ------------------------------------------------------- |
| `std`                    | Shared standardized schema (default for`/api/data`)   |
| `deepprime`              | DeepPrime native columns                                |
| `pridict` / `pridict2` | PRIDICT native columns                                  |
| `oped`                   | OPED native columns (`Target(47bp)`, `PBS`, `RT`) |

Model-format conversion is owned by **PE Database** (`GET /api/filter?format=…`). PE Ensemble proxies the same contract at `GET /data/filter` and uses it for training and evaluation.

## PE Database API (overview)

| Method | Path                | Description                                                |
| ------ | ------------------- | ---------------------------------------------------------- |
| GET    | `/api/studies`    | List studies                                               |
| GET    | `/api/datasets`   | List datasets                                              |
| GET    | `/api/datasheets` | List datasheet catalog entries                             |
| GET    | `/api/scaffolds`  | List pegRNA scaffolds                                      |
| GET    | `/api/data`       | Load standardized rows for one datasheet                   |
| GET    | `/api/filter`     | Filter catalog and/or export model-format data with splits |
| GET    | `/api/statistics` | Aggregate edit statistics                                  |
| POST   | `/api/export`     | Re-export and/or re-standardize                            |
| POST   | `/api/convert`    | Standardize one sheet                                      |
| GET    | `/health`         | Health check                                               |

### Examples

```bash
# Catalog browse
curl "http://localhost:8000/api/studies"
curl "http://localhost:8000/api/datasets?study=deepprime"

# Standardized rows for one datasheet
curl "http://localhost:8000/api/data?study=deepprime&dataset=deepprime-clinvar&cell_line=HEK293T&pe_system=PE2max"

# Export DeepPrime-format training data with an 80/20 holdout split
curl "http://localhost:8000/api/filter?format=deepprime&study=pridict1&dataset=library2&cell_line=HEK293T&pe_system=PE2max&split_strategy=holdout_2&train_pct=0.8&test_pct=0.2"
```

## PE Ensemble API (overview)

| Method | Path                          | Description                         |
| ------ | ----------------------------- | ----------------------------------- |
| GET    | `/models`                   | List supported models               |
| GET    | `/models/{name}/weights`    | List registered weight sets         |
| GET    | `/data/filter`              | Proxy to PE-DB filter/export        |
| POST   | `/evaluate`                 | Queue an asynchronous benchmark job |
| GET    | `/evaluate/status/{job_id}` | Benchmark job status and metrics    |
| GET    | `/evaluate/logs/{job_id}`   | Benchmark job logs                  |
| GET    | `/evaluate/jobs`            | List recent benchmark jobs          |
| POST   | `/train`                    | Queue an asynchronous training job  |
| GET    | `/train/status/{job_id}`    | Job status and result               |
| GET    | `/train/logs/{job_id}`      | Incremental training logs           |
| GET    | `/train/jobs`               | Recent jobs                         |
| GET    | `/devices`                  | Available compute devices           |
| GET    | `/train/devices`            | Per-device queue occupancy          |
| POST   | `/predict`                  | Prediction endpoint (stub)          |
| GET    | `/health`                   | Health check                        |

Models: **DeepPrime**, **PRIDICT2**, **OPED**. Training supports per-device queuing (CUDA, MPS, CPU, …). See `services/pe-ensemble/README.md` and `services/pe-ensemble/jobs/README.md` for CLI and SLURM usage.

## PE Hub

Single-site UI with live backend health indicators:

- **Database** — catalog browser, multi-filter export to CSV
- **Ensemble** — model evaluation, training job monitor, ensemble tool, inline API docs

Environment variables (`.env` in `pe-hub/`):

| Variable                  | Default                   |
| ------------------------- | ------------------------- |
| `VITE_PE_DB_URL`        | `http://localhost:8000` |
| `VITE_ENSEMBLE_API_URL` | `http://localhost:8001` |

## Work with CLI on Remote Cluster

We recognize that a lot of CLI usage would be done on remote clusters. 

## Shared package: pe-common

```python
from pe_common import DATA_ROOT, MODEL_ROOT, DEVICE
from pe_common.devices import list_devices, resolve_device
from pe_common.splits import SplitConfig, assign_splits
from pe_common.sequence_utils import align_wt_mut_sequences
from pe_common.features import calculate_gc_content  # lazy-loaded
from pe_common import run_supervised_training_loop   # lazy-loaded (requires torch)
```

Install: `pip install -e packages/pe-common`. Details in `packages/pe-common/README.md`.

## Development

```bash
make test           # pytest
make format         # black
make lint           # flake8
```

Re-export or re-standardize data:

```bash
PE_DB_FORCE_EXPORT=1 ./scripts/start-pe-db-backend.sh
# or
curl -X POST 'http://localhost:8000/api/export?force_standardize=true'
```

## Contributing data

Although I am trying my best to scour the internet for all the relevant data, I am sure there are many studies that I have missed. If you have data that you would like to contribute to the database, please convert it to the format specified below and submit a pull request.

### Catalog metadata

Register the study and dataset(s) in [`services/pe-db/app/catalog/studies.py`](services/pe-db/app/catalog/studies.py), and place raw source files under `datasets/raw/<study>/`.

**Study**

- `key` — short identifier (e.g. `deepprime`)
- `display_name` — human-readable study name
- `publication_date` — publication date
- `authors` — citation authors string

**Dataset** (one or more per study)

- `name` — dataset name within the study
- `description` — short description of the screen / validation set
- `pegRNA_delivery_method` / `pe_delivery_method` — how each component was delivered
- `edit_scope` — `on_target` or `off_target`
- `experimental_method` — `in_vitro` or `in_vivo`
- `target_context` — `endogenous` (native chromosomal locus) or `non_endogenous` (synthetic reporter / cassette)
- `standardizable` — `True` when rows can be fully converted to the shared schema below

Each datasheet is identified by `{cell_line}-{pe_system}` under the dataset (e.g. `hek293t-pe2`).

### Standardized edit format (PE core)

Contributed edit-level tables should use the shared standardized columns (parquet or CSV). All geometry positions are **0-based, half-open** `[left, right)` within `wt_sequence` / `mut_sequence` (sequences may be padded with `N` so WT and Mut share length).

| Column                                     | Type  | Description                                              |
| ------------------------------------------ | ----- | -------------------------------------------------------- |
| `group_id`                               | int   | Identifier for a unique protospacer within the datasheet |
| `type_sub` / `type_ins` / `type_del` | bool  | Intended edit class (mutually exclusive)                 |
| `edit_len`                               | int   | Edit length (bp)                                         |
| `wt_sequence`                            | str   | Wild-type target-strand sequence                         |
| `mut_sequence`                           | str   | Edited target-strand sequence                            |
| `protospacer_location_l` / `_r`        | int   | Protospacer interval in the sequences                    |
| `pbs_location_l` / `_r`                | int   | PBS interval                                             |
| `rtt_location_l` / `_r`                | int   | Reverse-transcriptase template interval                  |
| `lha_location_l` / `_r`                | int   | Left homology arm interval                               |
| `rha_location_l` / `_r`                | int   | Right homology arm interval                              |
| `spcas9_score`                           | float | Optional SpCas9 / DeepSpCas9 score (`NaN` if unknown)  |
| `editing_efficiency`                     | float | Measured prime-editing efficiency                        |
| `original_fold`                          | float | Optional source train/val/test fold id (`NaN` if none) |

### Endogenous extension (chromosomal edits)

If `target_context` is `endogenous` (edits measured at native chromosomal locations), also include these columns. Values may be null when a field is unknown; the columns themselves should still be present.

| Column                        | Type | Description                                                                       |
| ----------------------------- | ---- | --------------------------------------------------------------------------------- |
| `endo_genome_build`         | str  | Assembly, e.g.`hg38`, `mm39`                                                  |
| `endo_chr`                  | str  | Chromosome                                                                        |
| `endo_start` / `endo_end` | int  | 0-based half-open genomic interval                                                |
| `endo_strand`               | int  | `+1` or `-1` (null if unknown)                                                |
| `endo_coord_ref`            | str  | What the interval anchors, e.g.`protospacer`, `variant`, `trip_integration` |
| `endo_coord_source`         | str  | Provenance string for the coordinates                                             |
| `endo_locus_id`             | str  | Optional convenience label (gene, barcode, site name, …)                         |

Gene / chromatin / expression annotations are **not** stored here; they can be recovered later from coordinates plus cell-line context.

## Citation

If you found this data repo useful in your study, please consider citing our publication:

# PE-DB: Prime Editing Database and Model Ensemble

A platform for prime editing efficiency data management, model evaluation, and training.

## Components

| Component | Port | Role |
|-----------|------|------|
| **PE Database** (`services/pe-db`) | 8000 | Catalog metadata (SQLite) + standardized edit records (parquet/CSV) |
| **PE Ensemble** (`services/pe-ensemble`) | 8001 | Model wrappers, training jobs, evaluation, weight registry |
| **PE Hub** (`pe-hub`) | 5173 | Unified React UI for catalog browsing, export, training, and evaluation |

Shared Python utilities live in `packages/pe-common`.

## Quick start

From the repository root:

```bash
./start-all.sh --install   # first time: install Python + npm deps
./start-all.sh
```

- PE Hub: http://localhost:5173
- PE Database API docs: http://localhost:8000/docs
- PE Ensemble API docs: http://localhost:8001/docs

### Manual setup

```bash
make setup && source venv/bin/activate
make install          # editable installs for root, pe-common, pe-ensemble
bash setup.sh         # optional legacy dataset prep scripts
```

Run backends individually:

```bash
# PE Database
cd services/pe-db
pip install -r requirements.txt
pip install -e ../../packages/pe-common --no-deps
uvicorn app.main:app --reload --port 8000

# PE Ensemble
cd services/pe-ensemble
pip install -e .
PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8001

# PE Hub
cd pe-hub && npm install && npm run dev
```

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
├── start-all.sh              # Start all three services
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

| Format | Use |
|--------|-----|
| `std` | Shared standardized schema (default for `/api/data`) |
| `deepprime` | DeepPrime native columns |
| `pridict` / `pridict2` | PRIDICT native columns |
| `oped` | OPED native columns (`Target(47bp)`, `PBS`, `RT`) |

Model-format conversion is owned by **PE Database** (`GET /api/filter?format=…`). PE Ensemble proxies the same contract at `GET /data/filter` and uses it for training and evaluation.

## PE Database API (overview)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/studies` | List studies |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasheets` | List datasheet catalog entries |
| GET | `/api/scaffolds` | List pegRNA scaffolds |
| GET | `/api/data` | Load standardized rows for one datasheet |
| GET | `/api/filter` | Filter catalog and/or export model-format data with splits |
| GET | `/api/statistics` | Aggregate edit statistics |
| POST | `/api/export` | Re-export and/or re-standardize |
| POST | `/api/convert` | Standardize one sheet |
| GET | `/health` | Health check |

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

| Method | Path | Description |
|--------|------|-------------|
| GET | `/models` | List supported models |
| GET | `/models/{name}/weights` | List registered weight sets |
| GET | `/data/filter` | Proxy to PE-DB filter/export |
| POST | `/evaluate` | Queue an asynchronous benchmark job |
| GET | `/evaluate/status/{job_id}` | Benchmark job status and metrics |
| GET | `/evaluate/logs/{job_id}` | Benchmark job logs |
| GET | `/evaluate/jobs` | List recent benchmark jobs |
| POST | `/train` | Queue an asynchronous training job |
| GET | `/train/status/{job_id}` | Job status and result |
| GET | `/train/logs/{job_id}` | Incremental training logs |
| GET | `/train/jobs` | Recent jobs |
| GET | `/devices` | Available compute devices |
| GET | `/train/devices` | Per-device queue occupancy |
| POST | `/predict` | Prediction endpoint (stub) |
| GET | `/health` | Health check |

Models: **DeepPrime**, **PRIDICT2**, **OPED**. Training supports per-device queuing (CUDA, MPS, CPU, …). See `services/pe-ensemble/README.md` and `services/pe-ensemble/jobs/README.md` for CLI and SLURM usage.

## PE Hub

Single-site UI with live backend health indicators:

- **Database** — catalog browser, multi-filter export to CSV
- **Ensemble** — model evaluation, training job monitor, ensemble tool, inline API docs

Environment variables (`.env` in `pe-hub/`):

| Variable | Default |
|----------|---------|
| `VITE_PE_DB_URL` | `http://localhost:8000` |
| `VITE_ENSEMBLE_API_URL` | `http://localhost:8001` |

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
PE_DB_FORCE_EXPORT=1 ./start-pe-db-backend.sh
# or
curl -X POST 'http://localhost:8000/api/export?force_standardize=true'
```

## Contributing data

Although I am trying my best to scour the internet for all the relevant data, I am sure there are many studies that I have missed. If you have data that you would like to contribute to the database, please convert it to the format specified below and submit a pull request.

### Contribution format

To start with, the metadata of the study should be included in the pull request, containing the following information for advanced search and filtering:

- `Study`: The name of the study that the data originated from
- `Published Time`: The time that the study was published, in YYYYMM format

For each dataset, you should indicate:

- `PE System`: The version of the prime editor used in the study
- `Cell Line`: The cell line used in the study
- `Dataset Type`: The type of study, which can be either `Library`(0), `Off-target`(1), `Endogenous`(2)

The data should be in the form of a csv file, containing the following columns:

- `WT Sequence`: The wild type sequence of the target loci
- `MT Sequence`: The mutated sequence of the target loci after prime editing
- `protospacer Location`: The relative index of the pegRNA in the WT and MT sequence, in the format of `start-end`, both inclusive
- `PBS Location`: The relative index of the PBS in the WT and MT sequence
- `RT Location WT`: The relative index of the RT in the WT sequence, note that this would be differet from the MT sequence if there is an insertion or deletion
- `RT Location MT`: The relative index of the RT in the MT sequence
- `Efficiency`: The efficiency of the prime editing, which is the percentage of the MT sequence in the total sequence

The rest of the columns are optional, but can be included if available:

- `Chromatin State`: The chromatin state of the target loci
- `Indel Percentage`: The percentage of indels in the total sequence

To add a new study to the catalog, also register it in `services/pe-db/app/catalog/studies.py` and add raw files under `datasets/raw/<study>/`.

## Citation

If you found this data repo useful in your study, please consider citing our publication:

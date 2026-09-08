# PE-DB: Prime Editing Database and Model Ensemble

A platform for prime editing efficiency data management, model evaluation, and training.

## Components

| Component                                        | Port | Role                                                                    |
| ------------------------------------------------ | ---- | ----------------------------------------------------------------------- |
| **PE Database** (`services/pe-db`)       | 8000 | Catalog metadata (SQLite) + standardized edit records (parquet/CSV)     |
| **PE Ensemble** (`services/pe-ensemble`) | 8001 | Model wrappers, training jobs, evaluation, weight registry              |
| **PE Hub** (`pe-hub`)                    | 5173 | Unified React UI for catalog browsing, export, training, and evaluation |

Shared Python utilities live in `packages/pe-common`.

## Documentation index

New to the codebase? Read [`docs/architecture.md`](docs/architecture.md) — it maps
every module to what it does. For a command cheat sheet, see [`QUICKREF.md`](QUICKREF.md).

**Orientation**

| Document                                        | What it covers                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------- |
| [`QUICKREF.md`](QUICKREF.md)                   | One-page cheat sheet: URLs, start commands, make targets, troubleshooting |
| [`docs/architecture.md`](docs/architecture.md) | Code map for every service and package, plus the data and training flows  |
| This file                                       | Install, project structure, API overview, data contribution guide         |

**Services and packages**

| Document                                                            | What it covers                                                                           |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`services/pe-db/README.md`](services/pe-db/README.md)             | Catalog schema, the export/standardize pipeline,`pedb` CLI, filter and split semantics |
| [`services/pe-ensemble/README.md`](services/pe-ensemble/README.md) | Model wrappers, training/tuning/evaluation APIs,`peen` CLI, hyperparameters            |
| [`packages/pe-common/README.md`](packages/pe-common/README.md)     | Shared constants, splits, devices, sequence helpers, training utilities                  |
| [`pe-hub/README.md`](pe-hub/README.md)                             | Frontend setup, pages, environment variables                                             |

**Data**

| Document                                                                                  | What it covers                                                      |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [`datasets/README.md`](datasets/README.md)                                               | Directory layout for raw / exported / standardized / formatted data |
| [Standardized edit format](#standardized-edit-format-pe-core)                              | The shared schema contributed data must use                         |
| [`txt/diagrams/illustration/database_er.mmd`](txt/diagrams/illustration/database_er.mmd) | Catalog ER diagram (Mermaid source)                                 |

**Models, weights, and training output**

| Document                                                                                                            | What it covers                                                        |
| ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`services/pe-ensemble/weights/README.md`](services/pe-ensemble/weights/README.md)                                 | Weight-set layout, ID conventions, manifest fields, DVC guidance      |
| [`services/pe-ensemble/jobs/README.md`](services/pe-ensemble/jobs/README.md)                                       | Filesystem job state, logs, retention, and where each job kind writes |
| [`services/pe-ensemble/config/training_presets/README.md`](services/pe-ensemble/config/training_presets/README.md) | Preset YAML schema and how hyperparameters resolve                    |
| [`vendor/models/README.md`](vendor/models/README.md)                                                               | Vendor submodules, PyTorch unification, per-model quirks              |

**Plugins (adding your own model)**

| Document                                                                | What it covers                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [`plugins/README.md`](plugins/README.md)                               | Authoritative plugin guide: layout, manifest, contracts, activation |
| [`docs/plugin-author-llm-prompt.md`](docs/plugin-author-llm-prompt.md) | Copy-paste prompt for generating a plugin bundle                    |
| [`docs/add-new-model-plugins.md`](docs/add-new-model-plugins.md)       | Design document for the plugin system                               |
| [`plugins/_template/README.md`](plugins/_template/README.md)           | Template starting point                                             |

**Experiments, tuning, and clusters**

| Document                                                                                                      | What it covers                                             |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [`scripts/README.md`](scripts/README.md)                                                                     | What each script directory is for                          |
| [`scripts/experiments/README.md`](scripts/experiments/README.md)                                             | Experiment catalog, split protocols, base-model evaluation |
| [`scripts/hyperparameter/README.md`](scripts/hyperparameter/README.md)                                       | Optuna HPO runners and shared options                      |
| [`scripts/experiments/scratch-benchmark/README.md`](scripts/experiments/scratch-benchmark/README.md)         | Cross-model from-scratch benchmark matrix                  |
| [`scripts/experiments/datasheet-benchmark/README.md`](scripts/experiments/datasheet-benchmark/README.md)     | Single-datasheet nested Optuna benchmark                   |
| [`scripts/experiments/pridict2-reproduction/README.md`](scripts/experiments/pridict2-reproduction/README.md) | PRIDICT 2.0 transfer + ensemble reproduction               |
| [`scripts/cluster/oxford-arc/README.md`](scripts/cluster/oxford-arc/README.md)                               | Oxford ARC setup, DVC, job submission, monitoring          |
| [`testdata/vendor_eval/README.md`](testdata/vendor_eval/README.md)                                           | Vendor evaluation fixtures and how to regenerate them      |

**Legacy** — [`src/README.md`](src/README.md) records where the old `src/` code
moved. `src/` and `services/pe-ensemble/frontend/` contain no active code.

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

| Script                                      | What it does                                                        |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `./scripts/setup-python-env.sh --install` | Create/update Python 3.11 env**and** install project packages |
| `./scripts/install-clis.sh`               | Install packages only (env must already be active)                  |
| `./scripts/start-all.sh --install`        | Run`install-clis.sh` + `npm install`, then start all services   |

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
uvicorn pe_db.main:app --reload --port 8000

# PE Ensemble
cd services/pe-ensemble
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001

# PE Hub
cd pe-hub && npm install && npm run dev
```

Legacy dataset prep (optional): `bash scripts/setup.sh`

Vendor model source code is under `vendor/models/` (git submodules). Pretrained
weights are versioned in `services/pe-ensemble/weights/` — see
`services/pe-ensemble/weights/README.md`.

## Project structure

Module-by-module descriptions live in [`docs/architecture.md`](docs/architecture.md).

```
pe-hub/
├── packages/pe-common/       # Shared constants, splits, devices, training helpers
├── pe-hub/                   # Unified web UI (React + Vite)
├── services/
│   ├── pe-db/                # FastAPI catalog + data service
│   │   ├── app/
│   │   │   ├── catalog/      # Study/dataset/scaffold registries (seeded)
│   │   │   ├── converter.py  # Export + standardize orchestration
│   │   │   ├── db/           # SQLAlchemy catalog repository
│   │   │   └── utils/        # Standardization and model-format conversion
│   │   └── pe_db/            # `pedb` CLI + headless library
│   └── pe-ensemble/          # FastAPI model service
│       ├── app/
│       │   ├── models/       # Model wrappers + weight registry
│       │   ├── training/     # Training runner, Optuna tuning, job state
│       │   ├── evaluation/   # Benchmarking + train/test leakage checks
│       │   ├── ensemble/     # Multi-model prediction fusion
│       │   ├── plugins/      # Plugin upload, validation, activation
│       │   └── compute/      # Per-device job scheduler and job plumbing
│       ├── pe_ensemble/      # `peen` CLI + headless library
│       ├── config/           # Shipped hyperparameter presets
│       ├── jobs/             # Filesystem-backed training job state
│       └── weights/          # Registered pretrained + trained checkpoints
├── datasets/
│   ├── raw/                  # Original study files (Excel, CSV, …)
│   ├── exported/             # Normalized CSV per datasheet (generated)
│   ├── standardized/         # Parquet in shared schema (generated)
│   ├── formatted/            # Cached model-native columns (generated)
│   └── catalog/              # SQLite catalog DB (generated)
├── vendor/models/            # Third-party model code (submodules)
├── plugins/                  # User-contributed model plugins
├── docs/                     # Architecture map + plugin design docs
├── scripts/                  # Setup, experiments, HPO runners, cluster jobs
└── Makefile                  # install, test, lint, format
```

## Data pipeline

On PE Database startup, `initialize_database()` runs three steps:

1. **Seed** — create SQL tables; insert studies, datasets, and scaffolds from Python registries (`pe_db/catalog/`)
2. **Export** — write `datasets/exported/` from raw files; register **Datasheet** rows in the catalog
3. **Standardize** — write `datasets/standardized/` parquet from exported CSVs

Edit-level measurements are **not** stored in SQL. They are loaded with Pandas from parquet/CSV behind the API. Catalog tables (`study`, `dataset`, `scaffold`, `datasheet`) are described in [`services/pe-db/README.md`](services/pe-db/README.md) and [`txt/diagrams/illustration/database_er.mmd`](txt/diagrams/illustration/database_er.mmd).

Supported studies include DeepPrime, DeepPE, PRIDICT1, PRIDICT2, MinsePIE, and OptiPrime (see [`pe_db/catalog/studies.py`](services/pe-db/pe_db/catalog/studies.py)).

- **Some datasets are only partially standardizable.** `pridict1/endogenous`,
  `pridict2/trip_analysis`, `deepprime/deepprime_off_subpool`: their parquet files
  carry filter metadata but no sequence or coordinate columns, so they are readable
  via `/api/filter` but cannot be exported in a model format.

### Output formats

| Format                     | Use                                                               |
| -------------------------- | ----------------------------------------------------------------- |
| `std`                    | Shared standardized schema (use`format=std` on `/api/filter`) |
| `deepprime`              | DeepPrime native columns                                          |
| `pridict` / `pridict2` | PRIDICT native columns (both run the PRIDICT2 feature pipeline)   |
| `oped`                   | OPED native columns (`Target(47bp)`, `PBS`, `RT`)           |
| `optiprime`              | OptiPrime native RNA columns                                      |

Active plugins can register additional formats. Model-format conversion is owned by **PE Database** (`GET /api/filter?format=…`). PE Ensemble proxies the same contract at `GET /data/filter` and uses it for training and evaluation.

## PE Database API (overview)

| Method | Path                    | Description                                                |
| ------ | ----------------------- | ---------------------------------------------------------- |
| GET    | `/api/studies`        | List studies                                               |
| GET    | `/api/datasets`       | List datasets                                              |
| GET    | `/api/datasheets`     | List datasheet catalog entries                             |
| GET    | `/api/scaffolds`      | List pegRNA scaffolds                                      |
| GET    | `/api/filter`         | Filter catalog and/or export model-format data with splits |
| GET    | `/api/statistics`     | Aggregate edit statistics                                  |
| POST   | `/api/plugins/reload` | Reload plugin converters                                   |
| GET    | `/health`             | Health check                                               |

### Examples

```bash
# Catalog browse
curl "http://localhost:8000/api/studies"
curl "http://localhost:8000/api/datasets?study=deepprime"

# Standardized rows for one datasheet (via filter; admin export/convert is CLI-only)
curl "http://localhost:8000/api/filter?format=std&study=deepprime&dataset=deepprime-clinvar&cell_line=HEK293T&pe_system=PE2max&split_strategy=none"

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

The service also exposes `/tune` (Optuna hyperparameter search), `/ensemble`
(multi-model fusion) and plugin management routes; see
[`services/pe-ensemble/README.md`](services/pe-ensemble/README.md) for the full list.

Built-in models: **DeepPrime**, **PRIDICT2**, **OPED**, **OptiPrime**, plus any
active plugins. Training supports per-device queuing (CUDA, MPS, CPU, …). See
[`services/pe-ensemble/README.md`](services/pe-ensemble/README.md) for CLI usage,
[`services/pe-ensemble/jobs/README.md`](services/pe-ensemble/jobs/README.md) for
job state and artifacts, and
[`scripts/cluster/oxford-arc/README.md`](scripts/cluster/oxford-arc/README.md) for SLURM.

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

Most heavy training and tuning happens on a cluster, so both CLIs run fully
in-process — no HTTP server needed. `peen` talks to PE-DB through
`pe_db.library` directly when `PE_DB_URL` is unset, which means a SLURM job needs
only the conda environment and the dataset files.

```bash
# On the cluster, after ./scripts/setup-python-env.sh --install
conda activate pe-hub
peen train --model deepprime --dataset-name my-run \
    --study pridict1 --dataset library1 \
    --split-strategy holdout_3 --train-pct 0.7 --val-pct 0.15 --test-pct 0.15 \
    --device auto
```

Job state, logs and registered weights land on the filesystem under
`services/pe-ensemble/{jobs,weights}/`, so `peen jobs` and `peen logs` work
after the batch job exits. Large inputs and outputs are moved with DVC.

A worked setup for Oxford ARC — module loading, DVC remotes, `sbatch` templates,
walltime guidance and a smoke checklist — is in
[`scripts/cluster/oxford-arc/README.md`](scripts/cluster/oxford-arc/README.md).

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
make test                 # all suites (isolated pytest processes)
make test GROUP=smoke     # HTTP/CLI operation paths only
make test-list            # named groups (training, plugins, models, …)
./scripts/run-tests.sh training evaluation
make format
make lint
```

Use `./scripts/run-tests.sh` (or `make test`) so each suite gets its own
interpreter and PYTHONPATH. Pass one or more group names to run a functional
subset; `./scripts/run-tests.sh --list` prints the groups.

Re-export or re-standardize data:

```bash
PE_DB_FORCE_EXPORT=1 ./scripts/start-pe-db-backend.sh
# or, without HTTP:
pedb export --force-reexport
pedb standardize --force
```

## Contributing data

Although I am trying my best to scour the internet for all the relevant data, I am sure there are many studies that I have missed. If you have data that you would like to contribute to the database, please convert it to the format specified below and submit a pull request.

### Catalog metadata

Register the study and dataset(s) in [`services/pe-db/pe_db/catalog/studies.py`](services/pe-db/pe_db/catalog/studies.py), add exporters and standardizers in [`services/pe-db/pe_db/studies/<study>.py`](services/pe-db/pe_db/studies/) (`register_study(...)` plus one import in `load_studies()`), and place raw source files under `datasets/raw/<study>/`.

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
- `partial` — `True` for filter-only parquet (metadata, no model-format export)

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

If `target_context` is `endogenous` (edits measured at native chromosomal locations), the standardized row is still a **full PE-core record**: wild-type / mutant sequences, edit flags, and all geometry columns. Sequences sit on a **200 bp target-strand genomic window** with the 20-nt protospacer at offset 90 (90 bp of 5' context + spacer + 90 bp of 3' context), so later conversion can crop DeepPrime's 74-mer, OPED's 47-mer, or PRIDICT's author frame without inventing 3' DNA.

Also include these coordinate columns. Values may be null when a field is unknown; the columns themselves should still be present.

| Column                        | Type | Description                                                                       |
| ----------------------------- | ---- | --------------------------------------------------------------------------------- |
| `endo_genome_build`         | str  | Assembly, e.g.`hg38`, `mm39`                                                  |
| `endo_chr`                  | str  | Chromosome                                                                        |
| `endo_start` / `endo_end` | int  | 0-based half-open genomic interval                                                |
| `endo_strand`               | int  | `+1` or `-1` (null if unknown)                                                |
| `endo_coord_ref`            | str  | What the interval anchors, e.g.`protospacer`, `variant`, `trip_integration` |
| `endo_coord_source`         | str  | Provenance string for the coordinates                                             |
| `endo_locus_id`             | str  | Optional convenience label (gene, barcode, site name, …)                         |

Gene / chromatin / expression annotations are **not** stored here; they can be recovered later from coordinates plus cell-line context. Cached 200 bp `reference_window` strings live in the study loci JSON under `datasets/raw/` and are applied at standardize time (see `retrieve_endo_genomic_loci.py`). Rows without a cached window keep their author PE-core sequences.

Partial sheets (`pridict1/endogenous`, `pridict2/trip-analysis`) still lack recoverable WT/Mut sequences, so they stay filter-only until those designs can be reconstructed.

## Citation

If you found this data repo useful in your study, please consider citing our publication:

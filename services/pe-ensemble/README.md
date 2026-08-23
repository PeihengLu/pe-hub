# PE Ensemble Service

A FastAPI-based web service for evaluating, training, and creating ensembles of Prime Editing efficiency prediction models.

## Features

- Access to state-of-the-art PE prediction models (DeepPrime, PRIDICT, PRIDICT2, OPED)
- Conventional ML baselines (for example, XGBoost) alongside vendor models
- Model training and evaluation endpoints
- Ensemble prediction capabilities
- Integration with PE Database service for data retrieval

## Models Supported

- **DeepPrime** - CNN-GRU model for PE efficiency prediction
- **PRIDICT2** - Attention-based bidirectional LSTM model for PE efficiency prediction
- **OPED** - Transformer-based model for PE efficiency prediction

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /models` - List available models
- `POST /train` - Queue an asynchronous training job
- `GET /train/status/{job_id}` - Training job status
- `GET /train/logs/{job_id}` - Incremental training logs
- `GET /train/jobs` - List recent training jobs
- `POST /tune` - Queue an asynchronous Optuna hyperparameter tuning job
- `GET /tune/status/{job_id}` - Tuning job status
- `GET /tune/logs/{job_id}` - Tuning job logs
- `GET /tune/jobs` - List recent tuning jobs
- `GET /devices` - List compute devices (GPU, MPS, XPU, CPU, …)
- `GET /train/devices` - Per-device training occupancy and queue depth
- `GET /data/filter` - Proxy PE-DB filter/export (same contract as PE Hub export)
- `POST /predict` - Sequence prediction stub (design workflow will use this later)
- `POST /evaluate` - Queue an asynchronous benchmark job
- `GET /evaluate/status/{job_id}` - Benchmark job status and metrics
- `GET /evaluate/logs/{job_id}` - Benchmark job logs
- `GET /evaluate/jobs` - List recent benchmark jobs
- `POST /ensemble` - Queue an asynchronous multi-model ensemble job
- `GET /ensemble/methods` - List supported fusion methods (no retrain)
- `GET /ensemble/status/{job_id}` - Ensemble job status and metrics
- `GET /ensemble/logs/{job_id}` - Ensemble job logs
- `GET /ensemble/jobs` - List recent ensemble jobs

## Running Locally

```bash
cd services/pe-ensemble
pip install -e .
PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8001
```

The service will be available at http://localhost:8001 by default.

## Development

```bash
# Install dependencies
pip install -e .

# Run development server
PE_DB_URL=http://localhost:8000 uvicorn app.main:app --reload --port 8001
```

## Data format and conversion

Standardized → model-format conversion is owned by the **PE-DB** service
(`GET /api/filter?format=deepprime&cell_line=...&pe_system=...`). PE Ensemble
exposes the same contract in two places:

- **`GET /data/filter`** — direct proxy to PE-DB (same as PE Hub `exportFiltered`)
- **`POST /evaluate`** — fetches via that filter API, then runs metrics

Both accept multi-value filters (`cell_line=adv&cell_line=hela`, etc.). Model
wrappers expect data already in each model's **native** format and do not
convert standardized data themselves. This keeps `pe-common` free of
model-specific conversion logic.

## Weights registry

Pretrained vendor (and plugin) weights live under `services/pe-ensemble/weights/`
and are **tracked in git**. Locally trained weight sets use the same on-disk
layout but are **gitignored**; they are indexed in `local_registry.json` while
`registry.json` only lists git-tracked sources. Each weight set has a
`manifest.json`; trained runs get structured IDs like
`deepprime__hek293t-pe2max__20260608__a1b2c3`. Override the path with
`WEIGHTS_ROOT` only when mounting an external volume.

On older checkouts where weights still sit under `vendor/models`, bootstrap once
with `python -m app.models.migrate_weights`.

## Install (cluster / headless)

```bash
# Activate conda/venv first (not system Python)
pip install -e packages/pe-common
pip install -e services/pe-db
pip install -e "services/pe-ensemble[library]"
```

Or: `./scripts/install-clis.sh` (editable install of both CLIs + tab completion + usage summary; refuses Apple CLT Python). Reload completion with `conda deactivate && conda activate <env>`.
This installs the short CLI aliases **`pedb`** / **`peen`** (also `pe-db` / `pe-ensemble`).
The peen CLI always uses in-process `pe_db.library` (same code path as `pedb`), so training,
tuning, evaluation, and ensembling do not require a local PE-DB HTTP server.
The FastAPI web service always fetches PE-DB data over HTTP via `PE_DB_URL`.

## Training CLI and SLURM

Run training without the PE Ensemble HTTP server or PE Hub (logs and job state land under `jobs/`).

Active model plugins under `PLUGINS_ROOT` (default `<repo>/plugins`) are loaded automatically at CLI startup, same as the HTTP service. See [`../../plugins/README.md`](../../plugins/README.md) for how to prepare and activate a plugin for both the web UI and CLI.

### `peen` subcommands

| Command | Purpose |
|---------|---------|
| `peen train` | Train a model (queued via device scheduler by default) |
| `peen tune` | Optuna hyperparameter search (in-process by default; `--queue` for scheduler) |
| `peen evaluate` | Benchmark a registered weight set |
| `peen ensemble` | Fuse member predictions (`--combine mean`, `weighted_mean`, …) |
| `peen models` / `weights` / `methods` / `devices` | Registry and device listings |
| `peen jobs` / `logs` | Inspect queued or completed jobs |

`python -m app.train_models` and `python -m app.tune_models` remain as thin shims.

### Cluster / headless (in-process PE-DB)

Install both packages, prepare data once, then train without a PE-DB server:

```bash
pip install -e packages/pe-common
pip install -e services/pe-db
pip install -e "services/pe-ensemble[pe-db]"

pedb init   # seed, export, standardize (or skip if datasets/ already prepared)

peen train \
  --model deepprime \
  --dataset-name library2 \
  --dataset library2 \
  --cell-line HEK293T \
  --pe-system PE2max \
  --device cuda:0
```

### Example with architecture overrides

```bash
peen devices

peen train \
  --model deepprime \
  --dataset-name library2 \
  --dataset library2 \
  --cell-line HEK293T \
  --pe-system PE2max \
  --device cuda:0 \
  --dp-hidden-size 128 \
  --dp-num-layers 1 \
  --hyperparameters-json '{"epochs":5,"load_pretrained":false}' \
  --model-kwargs-json '{"pe_system":"PE2max","cell_type":"HEK293T"}' \
  --weights-root /scratch/$USER/pe-ensemble/weights \
  --jobs-root /scratch/$USER/pe-ensemble/jobs
```

Optional model architecture flags (merged into `hyperparameters`; see also `--hyperparameters-json`):

| Model | Flags |
|-------|--------|
| DeepPrime | `--dp-hidden-size`, `--dp-num-layers` |
| OPED | `--oped-embedding-size`, `--oped-ffn-dim`, `--oped-encoder-layers`, `--oped-nhead`, `--oped-dropout` |
| PRIDICT2 | `--pridict2-embed-dim`, `--pridict2-z-dim`, `--pridict2-num-hidden-layers`, `--pridict2-dropout` |

Example OPED training with a smaller architecture:

```bash
python -m app.train_models \
  --model oped \
  --dataset-name library2 \
  --dataset library2 \
  --oped-embedding-size 64 \
  --oped-ffn-dim 512 \
  --oped-encoder-layers 2 \
  --oped-nhead 4 \
  --hyperparameters-json '{"epoch_num":20,"lr":0.0003}'
```

Fine-tune from a registered checkpoint with `--pretrained-weights` (sets `load_pretrained=true`,
freezes the backbone, and skips architecture flags). DeepPrime always fine-tunes every ensemble
member in the checkpoint together:

```bash
python -m app.train_models \
  --model deepprime \
  --dataset-name library2 \
  --dataset library2 \
  --cell-line HEK293T \
  --pe-system PE2max \
  --pretrained-weights DeepPrime_base \
  --hyperparameters-json '{"epochs":5,"lr":0.0001}'
```

`POST /train` accepts `device` (`auto`, `cuda:0`, `mps`, `cpu`, …). Multiple jobs can run
concurrently on different devices; the service maintains a per-device queue so only one job
uses a given device at a time.

See [jobs/README.md](jobs/README.md) for queue-only / worker-step patterns.

List available weight sets via `GET /models/{model_name}/weights` or from Python:

```python
from app.models.model_factory import ModelFactory

# DeepPrime: IDs match the original variant directory names.
dp = ModelFactory.create_model("deepprime", pe_system="PE2max", cell_type="HEK293T")
print(dp.list_available_weights())          # ['DeepPrime_base', 'DP_variant_293T_PE4max_Opti_220728', ...]
metrics = dp.evaluate(test_df, weights="DeepPrime_base")

# PRIDICT2: compact IDs use '__' instead of '/'.
pr = ModelFactory.create_model("pridict2")
print(pr.list_available_weights())          # ['pridict1_1__exp_2023-08-25_20-55-53__run_2', ...]
metrics = pr.evaluate(test_df, weights="pridict1_1__exp_2023-08-25_20-55-53__run_2")

# Omit `weights` to evaluate the currently trained/loaded model.
metrics = dp.evaluate(test_df)
```

When `weights` is provided, `evaluate` loads that set first (overriding any
previously loaded model). When it is omitted and no model has been
trained/loaded, `evaluate` raises a `ValueError` listing how to proceed.

## PyTorch environment

DeepPrime, OPED, and PRIDICT2 were originally built against different PyTorch
versions, but they run together here in **one environment** pinned to
`torch>=2.0,<2.9`. The key enabler is that every model loads its weights as a
`state_dict` rather than a full pickled module, so the weights are independent
of the PyTorch version. PRIDICT2 sets the practical floor (it requires
`torch>=2.0`); DeepPrime and OPED state_dicts load on any modern 2.x.

OPED's legacy full-pickle checkpoints are version-fragile and are **not** loaded
at runtime — the wrapper defaults to `pegRNA_Model_Merged_saved.order3_decoder_weights.pt`
and rejects full pickles. Use `python -m app.models.convert_oped_weights` to
regenerate the state_dict if needed. See `vendor/models/README.md` for details.

Verify the unified environment with:

```bash
pytest tests/test_weights_loading.py
```

## Dependencies

This service depends on:
- PE Database service for data retrieval
- Vendor model repositories under `vendor/models/` (each subdirectory is a separate model codebase)

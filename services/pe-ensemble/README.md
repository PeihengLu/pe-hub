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
- `POST /train` - Train a model
- `POST /predict` - Get predictions from a model
- `POST /evaluate` - Evaluate model performance
- `POST /ensemble` - Get ensemble predictions

## Running Locally

```bash
cd services/pe-ensemble
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001
```

The service will be available at http://localhost:8001 by default.

## Development

```bash
# Install dependencies
pip install -e .

# Run development server
PE_DB_URL=http://localhost:8000 uvicorn pe_ensemble.main:app --reload --port 8001
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

All pretrained and service-trained weights live under `services/pe-ensemble/weights/`
and are **tracked in git** (~630 MB of checkpoints). Each weight set has a
`manifest.json`; trained runs get structured IDs like
`deepprime__hek293t-pe2max__20260608__a1b2c3`. Override the path with
`WEIGHTS_ROOT` only when mounting an external volume.

On older checkouts where weights still sit under `vendor/models`, bootstrap once
with `python -m app.models.migrate_weights`.

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

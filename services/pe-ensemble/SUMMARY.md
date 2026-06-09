# PE Ensemble — Implementation Status

Concise snapshot of what exists today. Full usage docs: [README.md](README.md).

## Model wrappers

| Model | Wrapper | Weights |
|-------|---------|---------|
| DeepPrime | `app/models/deepprime_wrapper.py` | `weights/deepprime/<id>/` |
| OPED | `app/models/oped_wrapper.py` | `weights/oped/<id>/` |
| PRIDICT2 | `app/models/pridict2_wrapper.py` | `weights/pridict2/<id>/` |

All wrappers implement `pe_common.model_interface.BasePEModel`. Create via
`ModelFactory.create_model(name, device=..., **kwargs)`.

## API (FastAPI `app/main.py`)

- **Catalog** — `GET /models`, `GET /models/{name}/weights`
- **Data** — `GET /data/filter` (PE-DB proxy)
- **Evaluate** — `POST /evaluate` (PE-DB fetch or inline records; test split only)
- **Train** — `POST /train` with async job queue; `GET /train/status|logs|jobs`
- **Devices** — `GET /devices`, `GET /train/devices`
- **Predict** — `POST /predict` (stub)

## Training infrastructure

```
app/training/
  config.py           # Supported models and format mapping
  data.py             # PE-DB filter client
  device_scheduler.py # Per-device queue (one active job per device)
  jobs.py             # Filesystem job manifests under jobs/
  runner.py           # Training execution
  schemas.py          # TrainingRequest, split params
```

CLI: `python -m app.train_models` (see [jobs/README.md](jobs/README.md)).

## Tests

```
tests/
  test_model_wrappers.py
  test_weights_loading.py
  test_weights_registry.py
  test_device_scheduler.py
  test_training_jobs.py
```

Run from `services/pe-ensemble`: `pytest tests/ -v`

## Not yet implemented

- Full `POST /predict` response (currently returns a placeholder message)
- HTTP `POST /ensemble` endpoint (ensemble logic exists in `app/ensembler.py` for future use)

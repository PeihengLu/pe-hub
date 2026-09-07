# PE Ensemble — Implementation Status

Concise snapshot of what exists today. Full usage docs: [README.md](README.md).

## Model wrappers

| Model | Wrapper | Weights |
|-------|---------|---------|
| DeepPrime | `pe_ensemble/models/deepprime_wrapper.py` | `weights/deepprime/<id>/` |
| OPED | `pe_ensemble/models/oped_wrapper.py` | `weights/oped/<id>/` |
| PRIDICT2 | `pe_ensemble/models/pridict2_wrapper.py` | `weights/pridict2/<id>/` |

All wrappers implement `pe_common.model_interface.BasePEModel`. Create via
`ModelFactory.create_model(name, device=..., **kwargs)`.

## API (FastAPI `pe_ensemble/main.py`)

- **Catalog** — `GET /models`, `GET /models/{name}/weights`
- **Data** — `GET /data/filter` (PE-DB proxy)
- **Evaluate** — `POST /evaluate` (PE-DB fetch or inline records; test split only)
- **Ensemble** — `POST /ensemble` (multi-model fusion on test split; async job queue)
- **Train** — `POST /train` with async job queue; `GET /train/status|logs|jobs`
- **Devices** — `GET /devices`, `GET /train/devices`
- **Predict** — `POST /predict` (stub)

## Training infrastructure

```
pe_ensemble/training/
  config.py           # Supported models and format mapping
  data.py             # PE-DB filter client
  jobs.py             # Filesystem job manifests under jobs/
  runner.py           # Training execution
  schemas.py          # TrainingRequest, split params
```

CLI: `peen train` (see [jobs/README.md](jobs/README.md)).

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

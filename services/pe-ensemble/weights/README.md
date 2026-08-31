# PE Ensemble weights

Model weight sets used by the pe-ensemble service live here.

- **Vendor / plugin weights** are **versioned in this repository** and indexed in
  `registry.json`.
- **Locally trained weights** (`source: trained`) stay on disk for runtime use
  but are **gitignored**, indexed in `local_registry.json`. To version a trained
  set across laptop ↔ ARC: `dvc add` that directory (IDs
  `model__scope__YYYYMMDD__shortid` only; vendor blobs must stay in git).

Vendor pretrained weights were moved here from `vendor/models`; new weights from
`POST /train` are registered into the same tree under structured IDs.

## Layout

```
weights/
  registry.json          # git-tracked index (vendor + plugin)
  local_registry.json    # gitignored index (trained / local)
  deepprime/<id>/        # ensemble .pt files + mean.csv + std.csv + manifest.json
  oped/<id>/             # weights.pt + manifest.json
  pridict2/<id>/         # model_statedict/ + config/ + manifest.json
```

Override the location only when needed (e.g. Docker volume) via `WEIGHTS_ROOT`; the default is this directory.

## Weight set IDs

- **Vendor (migrated):** original names, e.g. `DeepPrime_base`, `pegRNA_Model_Merged_saved.order3_decoder_weights`, `pridict1_1__exp_2023-08-25_20-55-53__run_2`
- **Trained (local):** `<model>__<scope>__<YYYYMMDD>__<shortid>`, e.g. `deepprime__hek293t-pe2max__20260608__a1b2c3`

## Bootstrap (existing checkouts only)

If you have an older checkout where weights still live under `vendor/models`, run once:

```bash
cd services/pe-ensemble
python -m app.models.migrate_weights --dry-run   # preview
python -m app.models.migrate_weights             # move files and register
```

Fresh clones include vendor weights under this directory. Rebuild indexes after
adding manifests with:

```bash
python -c "from app.models import weights_registry; weights_registry.rebuild_index()"
```

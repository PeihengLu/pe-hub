# PE Ensemble weights

All model weight sets used by the pe-ensemble service live here and are **versioned in this repository** (not gitignored). Vendor pretrained weights were moved here from `vendor/models`; new weights from `POST /train` are registered into the same tree.

## Layout

```
weights/
  registry.json          # aggregate index (rebuildable from manifests)
  deepprime/<id>/        # ensemble .pt files + mean.csv + std.csv + manifest.json
  oped/<id>/             # weights.pt + manifest.json
  pridict2/<id>/         # model_statedict/ + config/ + manifest.json
```

Override the location only when needed (e.g. Docker volume) via `WEIGHTS_ROOT`; the default is this directory.

## Weight set IDs

- **Vendor (migrated):** original names, e.g. `DeepPrime_base`, `pegRNA_Model_Merged_saved.order3_decoder_weights`, `pridict1_1__exp_2023-08-25_20-55-53__run_2`
- **Trained:** `<model>__<scope>__<YYYYMMDD>__<shortid>`, e.g. `deepprime__hek293t-pe2max__20260608__a1b2c3`

## Bootstrap (existing checkouts only)

If you have an older checkout where weights still live under `vendor/models`, run once:

```bash
cd services/pe-ensemble
python -m app.models.migrate_weights --dry-run   # preview
python -m app.models.migrate_weights             # move files and register
```

Fresh clones should already include weights under this directory.

# Architecture and code map

How the repository fits together, and what each module is responsible for. Start
here when you need to find where a behaviour lives; use the per-directory
READMEs (linked from each section) for command-line usage.

- **Getting started / commands** — [`README.md`](../README.md), [`QUICKREF.md`](../QUICKREF.md)
- **This document** — where code lives and why
- **Full documentation index** — [README § Documentation index](../README.md#documentation-index)

## Contents

- [The three services](#the-three-services)
- [Data flow: raw files to model inputs](#data-flow-raw-files-to-model-inputs)
- [Training flow: request to registered weights](#training-flow-request-to-registered-weights)
- [Code map: `packages/pe-common`](#code-map-packagespe-common)
- [Code map: `services/pe-db`](#code-map-servicespe-db)
- [Code map: `services/pe-ensemble`](#code-map-servicespe-ensemble)
- [Code map: `pe-hub` (frontend)](#code-map-pe-hub-frontend)
- [Directories that are not source](#directories-that-are-not-source)

## The three services

| Service | Language | Owns |
|---|---|---|
| `services/pe-db` | Python / FastAPI | The catalog, the raw→standardized pipeline, and **all** model-format conversion |
| `services/pe-ensemble` | Python / FastAPI | Model wrappers, training, tuning, evaluation, ensembling, the weight registry |
| `pe-hub` | React / TypeScript | The single web UI for both services |

Two rules explain most of the layout:

1. **PE-DB owns data shape.** Converting standardized rows into DeepPrime /
   PRIDICT / OPED / OptiPrime columns happens only in
   `services/pe-db/pe_db/formats/`. PE-Ensemble asks for data in a
   model's native format and never converts standardized rows itself. This is
   why every wrapper's `prepare_data` validates its columns and raises with a
   "fetch from PE-DB" message rather than trying to fix the frame.
2. **Edit-level measurements are not in SQL.** The SQLite catalog holds only
   metadata (`study`, `dataset`, `scaffold`, `datasheet`). The measurements
   themselves live in parquet and are read with Pandas behind the API. See
   [`txt/diagrams/illustration/database_er.mmd`](../txt/diagrams/illustration/database_er.mmd)
   for the catalog ER diagram.

Both services are usable two ways: as an HTTP server, or in-process through a
headless library (`pe_db.library`, `pe_ensemble.library`) that the `pedb` and
`peen` CLIs wrap. Cluster jobs use the in-process path so they need no running
server.

## Data flow: raw files to model inputs

```
datasets/raw/<study>/            Original published files (Excel, CSV)
  │   export_original_data()             pe_db/pipeline/run.py → pe_db/studies/<name>.py
  ▼
datasets/exported/<study>/<dataset>/<cell_line>-<pe_system>.csv
  │                                      Per-study column renaming, one CSV per datasheet
  │   standardize_exported_data()        pe_db/pipeline/run.py → pe_db/studies/<name>.py
  ▼
datasets/standardized/<study>/<dataset>/<cell_line>-<pe_system>.parquet
  │                                      Shared schema, 0-based half-open coordinates
  │   format_registry.convert_standardized()   pe_db/format_registry.py → pe_db/formats/
  ▼
datasets/formatted/<format>/...          Cached model-native columns
  │                                      pe_db/formatted_cache.py
  ▼
GET /api/filter?format=…&split_strategy=…    Filtered + split-assigned model input
```

Three stages run on PE-DB startup via `initialize_database()`
(`pe_db/catalog/initialize.py`): **seed** the catalog from the Python registries,
**export** raw files, then **standardize** exported CSVs.

Things worth knowing about this pipeline:

- **The standardized schema is the contract.** All geometry columns
  (`protospacer_location_l/r`, `pbs_location_*`, `rtt_location_*`,
  `lha_location_*`, `rha_location_*`) are 0-based half-open `[left, right)`
  offsets into `wt_sequence` / `mut_sequence`. WT and Mut are padded with `N` so
  they share a length; `rtt_location_r` and `rha_location_r` index the **mutated**
  sequence while their `_l` counterparts index WT. Endogenous sheets keep this
  full PE-core layout on a ~200 bp spacer-centered genomic window (offset 90)
  plus `endo_*` coordinate columns. The full column list is in
  [README § Standardized edit format](../README.md#standardized-edit-format-pe-core).
- **Export is skipped per dataset, not per study.** `_missing_exported_datasets`
  compares the dataset registry against what is on disk, so a newly registered
  dataset is picked up without `force_reexport`.
- **Standardization failures do not abort the run** (this executes on service
  startup), but every failure is collected and re-reported in one summary at the
  end. Check for `Standardized N datasheet(s); M failed` in the logs.
- **Rows with no efficiency measurement are dropped here**, by
  `_drop_unmeasured_efficiency_rows`, so the standardized parquet defines which
  rows are trainable. It runs in `standardize_pe_data` *after* the per-study
  standardizers, because several of them attach columns positionally and would
  desynchronize if rows disappeared mid-way. A real `0.0` is kept; only missing
  or non-numeric labels go.
- **Some datasets are only partially standardizable.** `pridict1/endogenous`,
  `pridict2/trip_analysis` and `deepprime/deepprime_off_subpool` get
  filter-only parquets that lack sequence and coordinate columns, so they are
  reachable through `GET /api/filter` but not through `format=` exports.
  Fully standardizable endogenous sheets (DeepPE endo, MinSePIE, PRIDICT
  library2-invivo) emit the same PE-core columns as reporter screens, on a
  200 bp genomic window.
- **`pridict` and `pridict2` share one converter.** Both run the full PRIDICT2
  feature pipeline, including the ViennaRNA MFE features.

## Training flow: request to registered weights

```
POST /train  or  peen train
  │   pe_ensemble/main.py → pe_ensemble/training/schemas.py (request validation, incl. splits)
  ▼
pe_ensemble/training/jobs.py            create_job() → jobs/<job_id>/{request,manifest}.json
  │
  ▼
pe_ensemble/compute/device_scheduler.py Queue per device; one running job per device
  │
  ▼
pe_ensemble/training/runner.py          execute_training()
  │   1. fetch_training_dataframe()      pe_ensemble/training/data.py → PE-DB /api/filter
  │   2. exclude_test_partition()        pe_common/splits.py
  │   3. record training loci            → leak audit sidecar
  │   4. resolve_hyperparameters()       pe_ensemble/training/hyperparameter_presets.py
  │   5. model.train(...)                pe_ensemble/models/<model>_wrapper.py
  │   6. register_trained_model()        pe_ensemble/models/weights_registry.py
  ▼
weights/<model>/<weight_id>/    Weight files + manifest.json + train_target_loci.json
```

`peen tune` wraps the same `execute_training` per Optuna trial
(`pe_ensemble/training/tune_runner.py`), writes the best hyperparameters to a local
preset YAML, and optionally registers weights for the best trial.

### What the wrappers share, and where they differ

All three models run the same outer shell — fetch → exclude test → optional CV
loop → final fit through `pe_common.training.fit_lightning_module`. The
remaining differences are deliberate:

| Aspect | DeepPrime | OPED | PRIDICT2 |
|---|---|---|---|
| Epoch hyperparameter | `epochs` | `epoch_num` | `num_epochs` |
| Default LR scheduler | `none` | `step` | `none` |
| Frozen-except module (fine-tuning) | `model.head` | `fully_connected_layers` | `decoder` |
| `predict` input | dict of tensors | encoded DataFrame | DataFrame or DataLoader |
| Tuning objective | negative val loss | val Spearman | val Spearman |
| Needs disk round-trip to save | no | no | **yes** (vendor saver owns the layout) |

The epoch-name divergence is historical and mirrors each vendor's own
hyperparameter name; the search spaces in `pe_ensemble/training/search_spaces.py` use
the matching name per model. Wrappers read the documented aliases through
`first_hyperparam` (`pe_common.training`) and resolve `load_pretrained` /
`weights` through `pe_ensemble/models/hparams.py`. Lightning and JAX training loops
stay separate.

### Cross-cutting training behaviour

- **Runs are seeded.** `pe_common.training.seed_training_run` is called at the
  top of every wrapper's `train()`, before the model is constructed, so weight
  initialization, DataLoader shuffling and dropout are all reproducible. The
  seed comes from the `seed` (or `random_state`) hyperparameter and defaults to
  `DEFAULT_TRAINING_SEED = 42`. Pass `seed: null` to opt out. Seeding inside the
  fit alone is not enough — weights are created earlier.
- **DeepPrime regresses on `log1p(efficiency)`.** Vendor inference ends with
  `exp(pred) - 1`, so the checkpoints emit log-space values. `_to_model_space`
  applies the forward transform during training and `predict` inverts it.
  Validation correlations are reported in efficiency units.
- **Feature normalization travels with the checkpoint.** DeepPrime z-scores its
  tabular features, so a from-scratch run fits `mean`/`std` on the *training*
  split only and saves them (plus `architecture.json`) into the weight
  directory. Without this a reloaded model would be scored against DeepPrime's
  vendor statistics.
- **Missing labels are rejected, not imputed.** Standardization already drops
  unmeasured rows, and converters keep a missing label as NaN instead of `0.0`,
  so a NaN reaching a wrapper means the frame bypassed that pipeline. The
  wrapper raises rather than training on a fabricated zero.
- **Lightning checkpointing is off.** The history callback keeps the best
  `state_dict` in memory and the registry persists the final model, so the
  default `ModelCheckpoint` would only litter `<cwd>/checkpoints`.
- **Training records its own data provenance.** The universal target-locus IDs
  behind each run are written to `train_target_loci.json` so
  `pe_ensemble/evaluation/leakage.py` can detect train/test overlap later.

## Code map: `packages/pe-common`

Shared library, installed with `pip install -e packages/pe-common`. Imported by
both services. Usage: [`packages/pe-common/README.md`](../packages/pe-common/README.md).

| Module | Responsibility |
|---|---|
| `constants.py` | `DATA_ROOT`, `MODEL_ROOT`, `DEVICE` and other path/env anchors |
| `splits.py` | `SplitConfig`, `assign_splits`, group-aware holdout/CV assignment, `exclude_test_partition`, CV fold iteration |
| `training.py` | `fit_lightning_module` (the one Lightning trainer), seeding, LR schedulers, early stopping, `regression_metrics` |
| `devices.py` | Device discovery, `resolve_device`, Lightning accelerator mapping |
| `sequence_utils.py` | `align_wt_mut_sequences`, padding insert/remove, coordinate shifting |
| `data_utils.py` | Frame helpers and `TARGET_UID_COLUMN`, the universal locus key used for leak audits |
| `features.py` | MFE (ViennaRNA), melting temperature, GC content. Lazy-imported |
| `model_interface.py` | `BasePEModel` — wrappers implement train/eval/predict; default hooks cover OPED prepare and PRIDICT2 stderr |
| `plugins.py` | Plugin manifest parsing and discovery |
| `plugin_validation.py` | The validation harness that gates plugin activation |
| `cell_lines.py` | Cell-line name normalization |
| `filter_params.py` | Catalog/edit/split field lists shared by CLI, FastAPI Query models, request bodies, and the TypeScript client |
| `conversion_progress.py` | Progress reporting shared with PE-DB conversion |

`training.py` and `features.py` are loaded lazily because they pull in torch and
ViennaRNA; importing `pe_common` alone stays cheap.

## Code map: `services/pe-db`

Catalog and data service. Usage and API: [`services/pe-db/README.md`](../services/pe-db/README.md).

### `pe_db/` — installable package

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI catalog/filter/health routes (`/api/studies`, `/api/filter`, …) |
| `library.py` | Headless equivalent of the HTTP API; what `pedb` and in-process `peen` call |
| `cli.py` | `pedb` / `pe-db` console entry |
| `converter.py` | Orchestrates export → standardize → model-format conversion, with cache lookup |
| `format_registry.py` | Maps a format name (`std`, `deepprime`, `pridict`, `pridict2`, `oped`, `optiprime`, plus plugin formats) to its converter |
| `formatted_cache.py` | Revision-gated on-disk cache of converted frames |
| `loaders.py` | Reads standardized parquet; normalizes `-`/`_` in path segments |
| `plugin_loader.py` | Registers converters contributed by plugins |
| `process_pool.py` | Worker pool for the expensive MFE feature pass |
| `mfe_worker.py` | Spawn-safe PRIDICT2 MFE worker entry |
| `config.py` | Paths and environment flags |

### `pe_db/catalog/` — what data exists

| Module | Responsibility |
|---|---|
| `studies.py` | `STUDY_REGISTRY` / `DATASET_REGISTRY` |
| `records.py` | `StudyRecord`, `DatasetRecord` (`partial` = filter-only parquet) |
| `datasheets.py` | Scans `datasets/exported/` and indexes `Datasheet` rows; infers scaffolds |
| `scaffolds.py` | pegRNA scaffold sequences and IDs |
| `seed.py` | Writes the registries into SQL; migrates legacy columns |
| `initialize.py` | The startup sequence: seed → export → standardize |

Adding a study means catalog rows in `studies.py` plus a pipeline module under
`pe_db/studies/` — see [README § Contributing data](../README.md#contributing-data).

### `pe_db/pipeline/` and `pe_db/studies/` — the pipeline

| Module | Responsibility |
|---|---|
| `pipeline/run.py` | `export_original_data`, `standardize_exported_data`, `standardize_pe_data` |
| `pipeline/registry.py` | Study exporters/standardizers/scaffold callbacks |
| `pipeline/schema.py` | Shared standardized columns and builders |
| `studies/<name>.py` | Per-study export + standardize (no edits to the orchestrator) |

### `pe_db/formats/` — model-native converters

| Module | Responsibility |
|---|---|
| `common.py` | Standardized schema check and series helpers |
| `thermo.py` | Shared Tm / GC / ViennaRNA helpers |
| `pridict.py` | PRIDICT / PRIDICT2 features |
| `deepprime.py` | DeepPrime 74-mer + thermo features |
| `optiprime.py` | OptiPrime RNA-alphabet inputs |
| `oped.py` | OPED 47-bp target / PBS / RT |

`utils/convert_data.py` re-exports these for tests and the MFE process-pool worker.

### `pe_db/utils/` — scoring helpers

| Module | Responsibility |
|---|---|
| `deepspcas9.py` | Extracts 30-mer windows and fills missing SpCas9 scores (TensorFlow 1.x model) |
| `json_utils.py` | NaN/Inf-safe JSON encoding for API responses |

`pipeline/` plus `studies/` own export and standardization. To trace one dataset, find its `pe_db/studies/<name>.py` module.

### `pe_db/db/` — SQL layer

| Module | Responsibility |
|---|---|
| `repository.py` | Filtering, conversion dispatch, split assignment, merge handling |
| `models.py` | SQLAlchemy tables for the catalog |
| `schemas.py` | Pydantic response models |
| `session.py` | Engine and session lifecycle |
| `base.py` | Declarative base |

HTTP is `uvicorn pe_db.main:app`. Ensemble HTTP is `uvicorn pe_ensemble.main:app`.

## Code map: `services/pe-ensemble`

Model service. Usage and API: [`services/pe-ensemble/README.md`](../services/pe-ensemble/README.md).

### `pe_ensemble/` — FastAPI

`main.py` holds every HTTP route. Job create/submit/status/logs/kill and
catalog lookups go through `library.py`; plugin upload/validate still talks
to `plugins/manager.py`. `plugin_loader.py` imports plugin wrappers.
`train_models.py` and `tune_models.py` are thin script entry points.

### `pe_ensemble/models/` — wrappers and the weight registry

| Module | Responsibility |
|---|---|
| `pridict2_wrapper.py` | `PERNNDistributionModel` training path plus vendor PRIEML load for inference |
| `oped_wrapper.py` | k-mer tokenization and transformer training |
| `deepprime_wrapper.py` | Ensemble fine-tuning and from-scratch training |
| `optiprime_wrapper.py` | OptiPrime (JAX/Flax stack; no tuning search space) |
| `hparams.py` | Pretrained weight ID, evaluate() weight check, CV-fold shell |
| `weights_registry.py` | The single place weights are written, indexed, resolved and provenance-stamped |
| `registry.py` | `ModelSpec` catalog: the four built-in models plus active plugins |
| `model_factory.py` | Name → wrapper instance |
| `*_vendor_provenance.py` | Records which published checkpoint each vendor weight set came from |
| `migrate_weights.py` | One-off migration of vendor weights into the registry layout |
| `convert_oped_weights.py` | Converts OPED's pickled checkpoints to state dicts |
| `vendor_path.py` | Locates `vendor/models/` for imports |

Weight-set layout, ID conventions and manifest fields:
[`services/pe-ensemble/weights/README.md`](../services/pe-ensemble/weights/README.md).

### `pe_ensemble/training/` — training and tuning

| Module | Responsibility |
|---|---|
| `runner.py` | `execute_training` — fetch, wrapper hooks, train, register weights |
| `tune_study.py` | Optuna study lifecycle, preset writing, optional final train |
| `tune_runner.py` | One Optuna trial; extracts the objective metric per model |
| `search_spaces.py` | Per-model search spaces and objective metric names |
| `hyperparameter_presets.py` | Merges baselines + shipped YAML + local YAML + request overrides |
| `data.py` | Builds PE-DB filter params and fetches the training frame |
| `jobs.py` / `tune_jobs.py` | Domain manifests; storage is `JobStore` |
| `dataset_key.py` | Canonical preset lookup keys for merged/multi-dataset filters |
| `progress_log.py` | Epoch log lines, stdout/stderr tee, cancellation hooks |
| `schemas.py` | Request models, including split validation |
| `config.py` | Job/study roots and model aliases |
| `pe_db_access.py` | In-process vs HTTP PE-DB access |
| `conversion_progress.py` | Streams PE-DB conversion progress into job logs |
| `model_baselines.py` | Code-level hyperparameter fallbacks |
| `model_architecture.py` | UI architecture choice → hyperparameters |
| `tuning_schemas.py` | Tuning request and summary models |

Hyperparameters resolve in this order, later winning: `model_baselines.py` →
`config/training_presets/<model>.yaml` (shipped) →
`config/training_presets_local/<model>.yaml` (Optuna output, gitignored) →
the request's `hyperparameters`. `hyperparameter_mode: "replace"` skips the YAML
layers entirely, which is what tuning trials use.

### `pe_ensemble/compute/` — scheduling and job plumbing

| Module | Responsibility |
|---|---|
| `device_scheduler.py` | Per-device queues; kind → `JobStore` + execute table |
| `job_store.py` | Shared filesystem job registry (create/list/logs/status) |
| `job_lifecycle.py` | Kill and delete semantics |
| `job_logging.py` | Routes the root logger into a job's log file |
| `manifest_io.py` | Atomic JSON writes and truncation-tolerant reads |
| `job_cancel.py` | In-memory cancellation flags |

### `pe_ensemble/evaluation/`, `pe_ensemble/ensemble/`, `pe_ensemble/plugins/`

- `evaluation/` — `runner.py` evaluates on the test partition only, using
  wrapper `prepare_evaluation_frame` / stderr-capture hooks;
  `leakage.py` compares a weight set's recorded training loci against the
  evaluation set and warns or excludes; `benchmark.py` resolves named
  benchmarks.
- `ensemble/` — `combine.py` fuses member predictions; `runner.py`
  orchestrates multi-model jobs via `predict_on_frame`.
- `plugins/` — `manager.py` handles upload, activation and removal;
  `validation_jobs.py` is a thin `JobStore` wrapper around validation manifests.

`training/`, `evaluation/`, `ensemble/`, and `plugins/` keep domain `create_job`
and `mark_succeeded` helpers; the shared filesystem mechanics live in
`pe_ensemble/compute/job_store.py`. On-disk layout is unchanged — see
[`services/pe-ensemble/jobs/README.md`](../services/pe-ensemble/jobs/README.md).

### `pe_ensemble/` — CLI

`cli.py` is the `peen` entry point (`train`, `tune`, `evaluate`,
`ensemble`, `jobs`, `logs`, `devices`, `models`). `main.py` is FastAPI.
Both call `pe_ensemble.library` for jobs, catalog, PE-DB filter, and
runners — the same split as `pedb` / `pe_db.main` and `pe_db.library`.

## Code map: `pe-hub` (frontend)

React + Vite single-page app, state-routed (no react-router). Setup:
[`pe-hub/README.md`](../pe-hub/README.md).

```
src/
├── App.tsx                  Top-level section switch (HubSection)
├── components/              Shared UI: HubNavbar, Card, ServiceGate, SplitAssignmentPanel, …
├── config/services.ts       Backend URLs from VITE_* env vars
├── context/                 ServiceHealthProvider — live backend health gating
├── pages/HomePage.tsx       Landing page
└── apps/
    ├── database/            Catalog browsing and export
    │   ├── pages/           CatalogPage, ExportPage
    │   ├── components/      ExportFilterBuilder, StatisticsCharts
    │   └── services/        peDbApi.ts
    └── ensemble/            Model workflows
        ├── pages/           BenchmarkPage, DesignPage, TrainingPage, EnsemblePage,
        │                    DocumentationPage, AddModelPage
        ├── components/      ComputeJobList, TrainingHyperparametersPanel, …
        ├── config/          Split params, model formats, scheduler defaults
        └── services/api.ts  PE-Ensemble client
```

Top-level sections are `home`, `database`, `ensemble` and `add-model`; within
`ensemble` the sub-pages are `benchmark`, `design`, `train`, `ensemble` and
`docs`. Filter/split query field names live once in
`apps/database/config/exportAttributes.ts` (`FILTER_LIST_FIELDS`,
`SPLIT_QUERY_FIELDS`) and must match `pe_common.filter_params`.

## Directories that are not source

| Path | Status |
|---|---|
| `src/` | **Empty legacy shell.** Only a README explaining where the code moved. No imports reference it |
| `services/pe-ensemble/frontend/` | **Retired.** Superseded by `pe-hub/` |
| `services/pe-db/build/` | **Stale build output.** Not the source of truth |
| `vendor/models/` | Third-party model code (git submodules) |
| `checkpoints/`, `artifacts/` | Scratch output, gitignored. Nothing in the service reads them |
| `datasets/`, `results/` | Data and experiment output; DVC-tracked, not in git |
| `txt/` | Manuscript sources and the diagrams under `txt/diagrams/` |
| `plugins/`, `testdata/plugins/` | Plugin bundles and test fixtures |

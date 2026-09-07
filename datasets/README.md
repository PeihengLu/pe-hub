# Datasets

Prime editing experimental data for the PE Database service.

## Directory layout

```
datasets/
├── raw/              # Original study files (checked in or restored locally)
│   ├── deepprime/
│   ├── pridict1/
│   ├── pridict2/
│   ├── minsepie/
│   └── deeppe/
├── exported/         # Generated CSV per datasheet (gitignored)
├── standardized/     # Generated parquet in shared schema (gitignored)
├── formatted/        # Generated model-format parquet cache (gitignored)
├── catalog/          # SQLite catalog DB pe_database.db (gitignored)
├── reference/        # hg38/mm39 FASTA (DVC; see datasets/reference.dvc)
└── dataprep/         # Legacy one-off prep scripts
```

`raw/` stays in **git**. Reference genomes are **DVC** (`dvc pull`) so they are
not in GitHub; the store is Oxford ARC (see
[`scripts/cluster/oxford-arc/README.md`](../scripts/cluster/oxford-arc/README.md#dvc-selective-artifacts)).

The PE Database service generates `exported/`, `standardized/`, and `catalog/` on
startup via `initialize_database()` (see
[`services/pe-db/README.md`](../services/pe-db/README.md)).

## Who writes each stage

Each directory has exactly one producer, so a wrong value can be traced to one
file:

| Stage | Written by | Trigger |
|---|---|---|
| `raw/` | Humans (committed) | — |
| `exported/` | `_export_<study>_datasheets` in `services/pe-db/app/utils/standardize_data.py` | Startup, per missing dataset, or `force_reexport` |
| `standardized/` | `_standardize_<study>_<dataset>` in the same file | Startup, or `force_standardize` |
| `formatted/` | converters in `services/pe-db/app/utils/convert_data.py` | On demand from `format=` requests, then cached |
| `catalog/` | `services/pe-db/app/catalog/seed.py` + `datasheets.py` | Startup |

Export is skipped **per dataset**, not per study, so registering a new dataset
picks it up on the next startup without a forced rebuild. Standardization
failures do not abort startup; they are summarized in one log line
(`Standardized N datasheet(s); M failed`).

### Unmeasured edits are dropped, not zero-filled

An edit with a blank efficiency cell was never measured in that cell line,
which is not the same as an efficiency of zero. Standardization drops those rows
and logs the count:

```
Dropped 2 of 33 row(s) from deeppe/deeppe_endo hek293t-pe2: no editing_efficiency measurement.
```

So `standardized/` is the definition of "rows you can train on", and a genuine
measurement of `0.0` is preserved. Converters no longer substitute `0.0` for a
missing label either, so if one ever reaches a model wrapper (from data that did
not come through this pipeline) training fails loudly instead of learning that
unmeasured edits are inefficient.

## Standardized format

The shared schema stores essential sequence and outcome fields for each edit.
Key columns (hyphenated in parquet):

| Column | Description |
|--------|-------------|
| `wt-sequence` | Target locus before editing |
| `mut-sequence` | Target locus after editing |
| `edit-efficiency` | Editing efficiency (fraction or percent per source) |
| `group-id` | Stable group key for train/test splits (same locus) |
| `edit-type` | `sub`, `ins`, or `del` |
| `edit-length` | Edit size |
| `original-fold` | Author train/test assignment when available |

Geometry columns (`pbs-location-l`, `rtt-location-r`, …) are **0-based half-open
offsets** into the padded sequences, not genomic coordinates. The full column list
and the padding convention are in
[root README § Standardized edit format](../README.md#standardized-edit-format-pe-core).

Model-specific columns are produced on demand by `GET /api/filter?format=…`
(`std`, `deepprime`, `pridict`, `pridict2`, `oped`, `optiprime`, plus any format
registered by an active plugin). Converted outputs are cached under
``formatted/{format}/…`` and cleared when data is rebuilt via ``force_reexport``
or ``force_standardize``.

## Studies

Registered in `services/pe-db/app/catalog/studies.py`:

| Study | Raw path | Notes |
|-------|----------|-------|
| `deepprime` | `raw/deepprime/` | DeepPrime benchmark sets |
| `pridict1` | `raw/pridict1/` | PRIDICT library and endogenous data |
| `pridict2` | `raw/pridict2/` | PRIDICT2 libraries and TRIP analysis |
| `minsepie` | `raw/minsepie/` | MinsePIE insert libraries |
| `deeppe` | `raw/deeppe/` | DeepPE benchmark sets |

Some datasets are **partially standardizable** (metadata-only conversion). See
`PARTIAL_STANDARDIZABLE_DATASETS` in `services/pe-db/app/utils/standardize_data.py`.

## Model-specific notes

### OPED

Inference expects three columns (produced by `format=oped` export):

- `Target(47bp)` — 47 bp wild-type window starting 4 bp upstream of the spacer
- `PBS` — reverse complement of the primer binding site
- `RT` — reverse complement of the reverse transcriptase template

All sequences are 5′→3′ on the edited strand.

### DeepPrime / PRIDICT

Use `format=deepprime`, `format=pridict`, or `format=pridict2` on `/api/filter`.
Column names match each vendor model's training scripts.

## Legacy prep

`datasets/dataprep/` contains older standalone scripts (`restore_pridict.py`,
`standarzied_data.py`). The supported path is PE Database startup export +
standardize. `bash scripts/setup.sh` still runs these for backward compatibility.

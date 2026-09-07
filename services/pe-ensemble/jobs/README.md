# Job state and training output

Every asynchronous job in PE Ensemble — training, tuning, evaluation, ensembling,
plugin validation — is persisted on the filesystem rather than in a database, so
a job survives a service restart and a SLURM batch job's results are readable
after it exits.

This directory (`jobs/`) holds **training** jobs. The other kinds use sibling
directories with the same layout.

## Layout

```
services/pe-ensemble/
├── jobs/<job_id>/              # peen train  /  POST /train
│   ├── request.json            # The TrainingRequest as submitted
│   ├── manifest.json           # Status, timings, device, weights_id, result
│   └── train.log               # Append-only log (what /train/logs streams)
├── tune_jobs/<job_id>/         # peen tune  /  POST /tune   (tune.log)
├── eval_jobs/<job_id>/         # peen evaluate  /  POST /evaluate
├── ensemble_jobs/<job_id>/     # POST /ensemble
├── validation_jobs/<job_id>/   # Plugin validation
├── tuning_studies/<study>.db   # Optuna SQLite storage (resumable)
└── weights/<model>/<id>/       # Registered checkpoints — see ../weights/README.md
```

All of these are gitignored. Roots are overridable by environment variable,
which is how the CLI isolates runs:

| Variable | Default |
|---|---|
| `TRAINING_JOBS_ROOT` | `services/pe-ensemble/jobs` |
| `TUNING_JOBS_ROOT` | `services/pe-ensemble/tune_jobs` |
| `TUNING_STUDIES_ROOT` | `services/pe-ensemble/tuning_studies` |
| `EVAL_JOBS_ROOT` | `services/pe-ensemble/eval_jobs` |
| `WEIGHTS_ROOT` | `services/pe-ensemble/weights` |

## Job manifest

`manifest.json` is the record the API reads for `/train/status/{job_id}`:

| Field | Meaning |
|---|---|
| `job_id`, `status` | `queued`, `running`, `stopping`, `succeeded`, `failed`, `cancelled`, `skipped` |
| `created_at`, `started_at`, `finished_at` | Timestamps (UTC, ISO 8601) |
| `model_name`, `dataset_source`, `dataset_name` | What was trained |
| `device_requested`, `device_assigned` | `auto` versus the device the scheduler picked |
| `weights_id`, `weights_label` | The registered weight set, when registration was enabled |
| `error` | Failure message, when `status` is `failed` |
| `result` | The full training payload, including the wrapper's own `result` dict |

Manifests are written through `pe_ensemble/compute/job_store.py` (atomic JSON via
`manifest_io.py`), so a reader never sees a half-written manifest. Plugin
validation jobs use the same store; they have no `request.json`.

`result` is intentionally untyped — each wrapper reports what is meaningful for
it. PRIDICT2 embeds a per-epoch history for every CV fold, so this field can get
large. The condensed version is what gets copied into the weight manifest's
`metrics` block.

## Logs

Logs are plain append-only text. `/train/logs/{job_id}?offset=N` returns
everything from a byte offset to EOF plus a `next_offset` for the next poll,
which is how the UI tails a running job.

There is **no rotation or size cap**. A long PRIDICT2 run with per-epoch logging
can produce a large `train.log`, and a client that polls from `offset=0` will
receive the whole file in one response.

## Retention

Nothing is pruned automatically. Deleting a job (`DELETE /train/jobs/{job_id}`)
removes its directory — manifest, request and log — but deliberately **keeps any
registered weights**, since those are the point of the run.

Consequences worth knowing:

- Job directories, tuning studies and local weight entries accumulate until you
  remove them.
- Every Optuna trial trains a model. Trials do not register weights, but
  PRIDICT2 trials each write intermediate artifacts to their own scratch
  directory (see below).
- Killing a job mid-run leaves its scratch directory behind.

To reclaim space, delete stale job directories and unwanted local weight entries,
then re-index the registry:

```bash
# Registered weights you no longer want
rm -rf services/pe-ensemble/weights/<model>/<weight_id>
python -c "import sys; sys.path.insert(0,'services/pe-ensemble'); \
from pe_ensemble.models import weights_registry; weights_registry.rebuild_index()"

# Finished job records (keeps weights)
rm -rf services/pe-ensemble/jobs/<job_id>
```

## Scratch directories

Most wrappers keep the trained model in memory and hand it to the weight
registry directly. PRIDICT2 is the exception: the vendor saver owns its
`model_statedict/` + `config/` layout, so the run is written to disk first and
then copied into the registry.

That scratch location defaults to a unique directory per run under the system
temp directory. Override the parent with `PRIDICT2_SCRATCH_ROOT`, or pin an
exact path with the `output_dir` hyperparameter. Avoid pointing concurrent jobs
at the same `output_dir`: they would overwrite each other's `final/` and `cv_*`
trees.

`checkpoints/` and `artifacts/` at the repository root are legacy scratch paths.
Nothing in the service reads them, and Lightning checkpointing is disabled, so
they should stay empty on new runs.

## Concurrency

The device scheduler (`pe_ensemble/compute/device_scheduler.py`) runs at most one job per
compute device and queues the rest, so two training jobs never contend for the
same GPU. Registry index rebuilds take a file lock, which matters when the API
server, the CLI and cluster jobs share one `WEIGHTS_ROOT`.

Note that `--device auto` only selects an accelerator. On a host with no
accelerator, request `cpu` explicitly.

## Related

- [`../weights/README.md`](../weights/README.md) — weight-set layout and manifests
- [`../README.md`](../README.md) — API and CLI reference
- [`../../../docs/architecture.md`](../../../docs/architecture.md) — how the training path fits together

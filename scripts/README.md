# Scripts

Setup helpers at the top level, plus three subdirectories for experiments,
hyperparameter search, and cluster jobs.

## Setup and service scripts

| Script | Purpose |
|---|---|
| `setup-python-env.sh` | Create/update the Python 3.11 conda env (or a venv with `--venv`); with `--install`, also runs `install-clis.sh` |
| `install-clis.sh` | Install `pe-common`, `pedb`, `peen`, vendor submodules and tab completion into the active env |
| `start-all.sh` | Start PE-DB, PE-Ensemble and the PE Hub frontend together; `--install` first runs `install-clis.sh` and `npm install` |
| `start-pe-db-backend.sh` | Start only PE-DB (honours `PE_DB_FORCE_EXPORT=1`) |
| `install-optiprime-deps.sh` | Install the OptiPrime / JAX stack (skippable with `SKIP_OPTIPRIME=1`) |
| `python-env.sh` | Shared env-detection helper sourced by the other setup scripts |
| `run-tests.sh` | Run all test suites in isolation; pass a group name (`smoke`, `training`, …) to select by function. `--list` prints groups. |
| `run-smoke-tests.sh` | Compatibility wrapper around `run-tests.sh` |
| `setup.sh` | Legacy dataset preparation; superseded by the startup pipeline |

Start here: [root README § Installation](../README.md#installation).

## Maintenance utilities

| Script | Purpose |
|---|---|
| `clear_cached_data.py` | Drop the formatted-data cache under `datasets/formatted/` |
| `regenerate_vendor_eval_fixtures.py` | Rebuild the CSV fixtures in `testdata/vendor_eval/` |

## Subdirectories

| Directory | Purpose |
|---|---|
| [`experiments/`](experiments/README.md) | Reproducible experiment runners: base-model evaluation, scratch benchmark, datasheet benchmark, PRIDICT2 reproduction |
| [`hyperparameter/`](hyperparameter/README.md) | Optuna HPO runners (`tune_hpo_cv5.sh`, `tune_hpo_holdout3.sh`) and shared option handling |
| [`cluster/oxford-arc/`](cluster/oxford-arc/README.md) | SLURM submission templates, environment bootstrap and DVC handling for Oxford ARC |

Shell scripts in `experiments/` and `hyperparameter/` source a `_common.sh` in
their own directory for shared defaults and environment-variable overrides; read
that file first when a flag's origin is unclear.

## Related

- [`../docs/architecture.md`](../docs/architecture.md) — what the scripts drive
- [root README § Documentation index](../README.md#documentation-index)

# Datasheet / dataset from-scratch benchmark

Nested Optuna benchmark for **one catalog dataset or datasheet**, optionally
filtered to provided edit types. Uses current **`pedb`** / **`peen`** libraries
(`pe_db.library.filter_data` and `pe_ensemble` tune + evaluate). No PE-DB HTTP
server.

## Protocol

`N` (`--n`) and `X` (`--n-trials` / `-x`) are always user-provided.

| Dataset size | Default (`--protocol auto`) | What runs |
|--------------|-----------------------------|-----------|
| **Small** (`n_rows < SIZE_THRESHOLD`, default 50 000) | N-fold **CV** | Outer fold *k* is test. Remaining loci are split train/val. **Optuna × X trials** on that fold. Retrain best HPs, score fold *k*. Repeat for all *N* folds. |
| **Large** (`n_rows >= SIZE_THRESHOLD`) | **holdout_3** × N | For each of *N* seeds: 70/15/15 group split, **Optuna × X trials**, retrain, score test. Seeds also seed model init. |

Override with `--protocol cv` or `--protocol holdout_3`.

Each Optuna trial trains **one** model (inner train/val), not an inner k-fold.
That is nested evaluation: hyperparameters are not shared across outer folds/seeds.

## Usage

```bash
conda activate pe-hub   # or pedb

# Datasheet + insertions only, auto protocol
./scripts/experiments/datasheet-benchmark/run.sh \
  --model pridict2 --n 5 -x 20 \
  --study minsepie --dataset library-insert-set12 \
  --cell-line hek293t --pe-system pe2 \
  --edit-type ins

# Whole dataset (all matching datasheets pooled, --merge on)
N=5 N_TRIALS=20 DEVICE=cuda:0 \
  ./scripts/experiments/datasheet-benchmark/run.sh \
  --model oped \
  --study pridict1 --dataset library1 \
  --edit-type sub --edit-type ins --edit-type del

# Force holdout_3 with 3 seeds even on a small sheet
PROTOCOL=holdout_3 ./scripts/experiments/datasheet-benchmark/run.sh \
  --model deepprime --n 3 -x 10 \
  --study deeppe --dataset deeppe-ht --edit-type sub

# Plan only (row count + protocol, no conversion / GPU)
python scripts/experiments/datasheet-benchmark/run_benchmark.py \
  --model pridict2 --n 5 -x 20 --dry-run \
  --study minsepie --dataset library-insert-set12 --edit-type ins

# One fold/seed (cluster fan-out)
python scripts/experiments/datasheet-benchmark/run_benchmark.py \
  --model pridict2 --n 5 -x 20 --index 0 \
  --study minsepie --dataset library-insert-set12 --edit-type ins
```

Smoke (mini `DATA_ROOT`, still pass N and X or use the env defaults from
`scripts/hyperparameter/_common.sh` when `SMOKE=1`):

```bash
SMOKE=1 DEVICE=cuda:0 N=2 N_TRIALS=1 \
  ./scripts/experiments/datasheet-benchmark/run.sh \
  --model pridict2 \
  --study minsepie --dataset library-insert-set12 --edit-type ins
```

## Outputs

Under `results/datasheet_benchmark/<RUN_ID>/` (gitignored via `/results`):

| File | Contents |
|------|----------|
| `plan.json` | Filters, protocol choice, aggregates |
| `results.jsonl` | One record per fold/seed (HPs, weights_id, test metrics) |
| `summary.csv` | Flat table |
| `summary_mean_std.csv` | Mean±std across successful repeats |
| `state/<repeat>.json` | Resume marker (`--skip-existing`) |

Latest run id: `results/datasheet_benchmark/LATEST_RUN_ID`.

## Notes

- **Edits:** `--edit-type sub|ins|del` (repeatable), plus the usual PE-DB filters
  (`--edit-length`, `--cell-line`, `--pe-system`, …).
- **Datasheet vs dataset:** a datasheet is `--study --dataset --cell-line --pe-system`.
  Omit cell/PE to pool every matching sheet (`--merge`, default on).
- `--skip-eval` registers best weights per fold/seed without running `peen evaluate`.
  Tune always writes `state/<repeat>.json` before eval so a walltime kill during
  eval can resume with `--skip-existing`.
- `--index` runs one fold/seed; `results.jsonl` is merged from `state/*.json`
  so parallel seed jobs do not clobber each other.
- Evaluation uses `--allow-data-leak` only because peen treats any **synthetic**
  in-domain holdout as `no_original_test_split`. Test loci are still excluded
  from training (`exclude_test_partition`).
- Models without an Optuna search space (e.g. OptiPrime) are rejected.

## Tests

```bash
pytest scripts/experiments/datasheet-benchmark/test_protocol.py
```

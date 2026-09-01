# Scratch benchmark (from-scratch model comparison)

Cross-model experiment for **DeepPrime**, **OPED**, and **PRIDICT2** on the same
benchmarks as `probe_scratch_train.sh`, with **Optuna tuning** and **holdout_3**
(70/15/15) splits. Designed for local runs and Oxford ARC batch submission.

Excludes OptiPrime.

## Matrix

| Benchmark | Study / dataset | Split |
|-----------|-----------------|-------|
| `pridict1-library1` | pridict1 / library1 | random holdout_3, seed 42 |
| `pridict2-library-diverse` | pridict2 / library-diverse | random holdout_3, seed 42 |
| `deepprime-clinvar` | deepprime / deepprime-clinvar | random holdout_3, seed 42 |

All cells use `--no-use-original-fold --split-random-state 42` (no author folds).

Models × benchmarks = **9 cells**. Each cell: tune → train (register weights) → evaluate on test.

## Local usage

```bash
conda activate pe-hub

# Smoke (mini DATA_ROOT, 1 trial, 2 epochs train)
SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh

# Full pipeline (sequential; long on ClinVar)
DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh

# Stages individually
./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
./scripts/experiments/scratch-benchmark/02_train_matrix.sh
./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh
```

### Filter to one model or benchmark

```bash
MODELS=oped BENCHMARKS=deepprime-clinvar \
  ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh

MODEL=pridict2 BENCHMARK=pridict1-library1 \
  ./scripts/experiments/scratch-benchmark/02_train_matrix.sh
```

### Resume / skip completed cells

```bash
SKIP_IF_DONE=1 ./scripts/experiments/scratch-benchmark/02_train_matrix.sh
```

State files: `scripts/experiments/scratch-benchmark/state/` (or under `/tmp/pe-hub-smoke-*` when `SMOKE=1`).

## ARC submission

From **htc-login** (not on the login node for compute):

```bash
cd $DATA/pe-hub
source scripts/cluster/oxford-arc/env.sh

# Dry-run scheduler validation
DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# Recommend submitting stages separately (tune is the long pole)
ARC_PARTITION=medium ARC_TIME=2-00:00:00 \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

ARC_PARTITION=medium ARC_TIME=1-00:00:00 \
  ./scripts/cluster/oxford-arc/submit.sh 02_train_matrix.sh

./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

# Single cell (1 GPU)
MODEL=oped BENCHMARK=deepprime-clinvar \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# All 9 cells in parallel (9 separate 1-GPU jobs — recommended if you have quota)
./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 01_tune_matrix.sh
./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 02_train_matrix.sh
./scripts/experiments/scratch-benchmark/submit_arc_matrix.sh 03_evaluate_matrix.sh
```

## Multi-GPU

**Single training/tuning job:** one GPU only. Lightning is configured with `devices=1` in
`pe_common.training`; extra GPUs on the same SLURM allocation stay idle.

**Parallel throughput:** submit one job per matrix cell (9 jobs × 1 L40S). Use
`submit_arc_matrix.sh` above, or manual `MODEL=… BENCHMARK=… submit.sh …` per cell.
Each job should keep `ARC_GPUS=1` and `DEVICE=cuda:0` (defaults in `env.sh`).

Bring artifacts home:

```bash
./scripts/cluster/oxford-arc/sync_from_arc.sh
cat scripts/experiments/scratch-benchmark/results/LATEST_RUN_ID
# summary CSV under results/<RUN_ID>/summary.csv
```

## Outputs

| Artifact | Location |
|----------|----------|
| Optuna DB | `services/pe-ensemble/tuning_studies/*.db` |
| Dataset presets | `services/pe-ensemble/config/training_presets_local/` |
| Trained weights | `services/pe-ensemble/weights/*__custom__*` |
| Pipeline state | `scripts/experiments/scratch-benchmark/state/` |
| Eval JSONL + CSV | `scripts/experiments/scratch-benchmark/results/<RUN_ID>/` |

## Tunable env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `N_TRIALS` | 20 | Optuna trials per cell |
| `SPLIT_RANDOM_STATE` | 42 | Reproducible holdout_3 |
| `BATCH_SIZE` | 128 | Final train batch size |
| `EARLY_STOPPING_PATIENCE` | 12 | Final train early stop |
| `MAX_EPOCHS_*` | 50 | Epoch cap per model (final train) |
| `NUM_WORKERS` | 15 | DataLoader workers |
| `DEVICE` | auto | CUDA device |

Smoke overrides (`SMOKE=1`): 1 trial, mini data locally; on ARC use `SMOKE=1` with full data via `submit.sh`.

## Comparison notes

- All models train **from scratch** (`load_pretrained=false`).
- Tuning optimizes **validation** metrics; **test** is only used in stage 03.
- PRIDICT2 uses `MSEloss` on `averageedited` (same as probe/reproduction).
- ClinVar is much slower than library1 (~260k rows); consider filtering benchmarks during development.

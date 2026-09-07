# Scratch benchmark (from-scratch model comparison)

Cross-model experiment for **DeepPrime**, **OPED**, and **PRIDICT2** on the same
pooled benchmarks as `evaluate_base_model_benchmarks.sh`. Each cell uses the
[`datasheet-benchmark`](../datasheet-benchmark/README.md) runner:

- **holdout_3** (70/15/15), **3 random split + init seeds**
- **10 Optuna trials per seed**
- Final train via `register_best_weights`, then test evaluation — **in the same job**

The **OptiPrime model** is excluded (no scratch HPO search space); **lib-mmr** /
**lib-cv** are included as datasets all three models train on.

## Matrix

| Benchmark | Study / dataset(s) | ~Rows | Split |
|-----------|-------------------|------:|-------|
| `pridict1-library1` | pridict1 / library1 | 92k | holdout_3 × 3 seeds |
| `pridict2-library-diverse` | pridict2 / library-diverse | 66k | holdout_3 × 3 seeds |
| `deeppe-pooled` | deeppe / ht + type + position + endo | 49k | holdout_3 × 3 seeds |
| `minsepie-insert-pooled` | minsepie / set12 + 18nt + codon-variant + codon-hek3 | 27k | holdout_3 × 3 seeds |
| `optiprime-lib-mmr` | optiprime / lib-mmr | 36k | holdout_3 × 3 seeds |
| `optiprime-lib-cv` | optiprime / lib-cv | 37k | holdout_3 × 3 seeds |
| `deepprime-clinvar` | deepprime / deepprime-clinvar | 289k | holdout_3 × 3 seeds |

All cells use `--protocol holdout_3` (no author folds). Base seed 42 → seeds 42, 43, 44.

Models × benchmarks × seeds = **63 jobs** (7 × 3 × 3). Each job: 10 HPO trials + 1 final train + eval.

## Local usage

```bash
conda activate pe-hub

# Smoke (mini DATA_ROOT, 1 trial, 2 seeds)
SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh

# Full pipeline (sequential; long on ClinVar)
DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh

# Stages individually
./scripts/experiments/scratch-benchmark/01_tune_matrix.sh   # HPO + train + eval
./scripts/experiments/scratch-benchmark/03_evaluate_matrix.sh  # aggregate (+ leftover eval)
```

### Filter to one model or benchmark

```bash
MODELS=oped BENCHMARKS=deepprime-clinvar \
  ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh

INDEX=0 MODEL=pridict2 BENCHMARK=pridict1-library1 \
  ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
```

### Resume / skip completed seeds

Optuna studies resume remaining trials (not 10 extra). Re-run 01 with the same
`RUN_ID` / `INDEX` after a walltime kill:

```bash
SKIP_IF_DONE=1 RUN_ID=<id> INDEX=0 MODEL=oped BENCHMARK=deepprime-clinvar \
  ./scripts/experiments/scratch-benchmark/01_tune_matrix.sh
```

`02_train_matrix.sh` is a thin alias for that resume (`SKIP_IF_DONE=1`).

State files: `scripts/experiments/scratch-benchmark/state/` (or under `/tmp/pe-hub-smoke-*` when `SMOKE=1`).

## ARC submission

From **htc-login** (not on the login node for compute). Defaults in
`env.sh.example`: **short / 12h / L40S**, one GPU.

Each short job is **one seed**: 10 Optuna trials + `register_best_weights` +
eval. That is 11 full holdout_3 trains, not a separate tune stage then train
stage. Smaller sheets (MinSePIE, DeepPE, lib-*, library-diverse, library1)
should fit 12h on L40S. ClinVar (~289k) may hit the wall; re-queue the same
`INDEX` with the same `RUN_ID` (Optuna continues, then final train + eval).

```bash
cd $DATA/pe-hub
source scripts/cluster/oxford-arc/env.sh

# Dry-run scheduler validation
DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# 63 jobs (7 × 3 × 3 seeds), short L40S
./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# One model × dataset → 3 seed jobs
MODEL=oped BENCHMARK=deepprime-clinvar \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# One seed only
INDEX=0 MODEL=oped BENCHMARK=deepprime-clinvar \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# Re-queue a timed-out seed (same RUN_ID)
RUN_ID=<id> SKIP_IF_DONE=1 INDEX=2 MODEL=oped BENCHMARK=deepprime-clinvar \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh

# Aggregate after GPU jobs finish (set RUN_ID from 01 logs / LATEST_RUN_ID)
RUN_ID=<id> ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh

# Legacy: one job per cell (all 3 seeds) — needs medium/48h
SUBMIT_SEEDS=0 ARC_PARTITION=medium ARC_TIME=2-00:00:00 \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh
```

## Multi-GPU

**Single training/tuning job:** one GPU only. Lightning is configured with `devices=1` in
`pe_common.training`; extra GPUs on the same SLURM allocation stay idle.

**Parallel throughput:** 63 jobs × 1 L40S (`submit.sh 01_tune_matrix.sh`).
Keep `ARC_GPUS=1` and `DEVICE=cuda:0`. Pack seeds with `SUBMIT_SEEDS=0` only if
you want fewer, longer jobs.

Save only runs you care about (see
[`scripts/cluster/oxford-arc/README.md`](../../cluster/oxford-arc/README.md#dvc-selective-artifacts)):

```bash
# on ARC
dvc add scripts/experiments/scratch-benchmark/results/<RUN_ID>
dvc add services/pe-ensemble/weights/<model>/<weights_id>
dvc push
git add '*.dvc' && git commit -m 'scratch-benchmark run' && git push

# on laptop
git pull && dvc pull
cat scripts/experiments/scratch-benchmark/results/LATEST_RUN_ID
```

## Outputs

| Artifact | Location |
|----------|----------|
| Optuna DB | `services/pe-ensemble/tuning_studies/*.db` |
| Dataset presets | `services/pe-ensemble/config/training_presets_local/` |
| Trained weights | `services/pe-ensemble/weights/*__custom__*` (one set per seed) |
| Pipeline state | `scripts/experiments/scratch-benchmark/state/` |
| Per-cell JSONL | `scripts/experiments/scratch-benchmark/results/<RUN_ID>/<model>__<bench>/` |
| Matrix summary | `scripts/experiments/scratch-benchmark/results/<RUN_ID>/summary.csv` |

## Tunable env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `N_TRIALS` | 10 | Optuna trials **per seed** |
| `N_SEEDS` | 3 | holdout_3 repeats (split + init seeds) |
| `PROTOCOL` | `holdout_3` | datasheet-benchmark protocol |
| `SPLIT_RANDOM_STATE` | 42 | Base seed (seeds are 42, 43, 44) |
| `INDEX` | unset | 0-based seed for cluster fan-out |
| `DEVICE` | auto | CUDA device |

Smoke overrides (`SMOKE=1`): 1 trial, 2 seeds, mini data locally; on ARC use `SMOKE=1` with full data via `submit.sh`.

## Comparison notes

- All models train **from scratch** (`load_pretrained=false`).
- Each seed has its own Optuna study and registered weights; test metrics are mean±std across seeds.
- PRIDICT2 uses `MSEloss` on `averageedited` (same as probe/reproduction).
- ClinVar (~289k rows) is the cell most likely to need a second short job.

# Oxford ARC — submit pe-hub tuning / training

> **Canonical toolkit:** shared, reusable copies of these scripts live in the
> sibling repo `oxford-arc` (`~/development/oxford-arc`). Prefer that checkout
> for new work and sharing. This directory remains for existing pe-hub
> workflows. Keep local `env.sh` gitignored — never commit usernames or emails.

# Official docs: [ARC User Guide](https://arc-user-guide.readthedocs.io/en/latest/)

# GPUs live only on the **htc** cluster.

## What to use

| Need                     | Choice                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Cluster                  | `htc` (`htc-login.arc.ox.ac.uk`)                                                                           |
| Login (on Uni net / VPN) | `ssh you@htc-login.arc.ox.ac.uk`                                                                             |
| Off-net                  | `ssh you@gateway.arc.ox.ac.uk` then hop to `htc-login`                                                     |
| Repo + data              | `$DATA/...` (project share, ~5 TiB). Avoid `$HOME` (15 GiB) for envs/datasets                              |
| Conda                    | `module load Anaconda3` (or Mamba); env **prefix** under `$DATA/envs/…` — see `env.sh` / `setup_interactive.sh` |
| Partitions               | `short` ≤12h · `medium` ≤48h · `long` ≤30d (default 1d unless `--time` set) · `devel` 10m test |
| GPU                      | `#SBATCH --gres=gpu:1` (+ optional `--constraint='gpu_sku:L40S'`)                                          |
| Build / pip / conda      | **interactive** node, not login: `srun -p interactive --gres=gpu:1 --pty bash`                         |

Co-investment GPU nodes are often limited to **short** (12h). Prefer ARC-owned L40S/A100 for **medium**/**long** HPO.

## One-time setup

1. Clone pe-hub under `$DATA` on **htc-login** (or clone locally and use the same
   workflow on both machines):

   ```bash
   cd $DATA
   git clone --recurse-submodules git@github.com:<org>/pe-hub.git pe-hub
   cd pe-hub
   ```

   If you already cloned without submodules:

   ```bash
   git submodule update --init --recursive
   ```

   Build datasets on ARC with `pedb init`. Sync reference genomes and other
   bulky artifacts with [rsync](#sync-artifacts-laptop-to-arc) from the laptop.
2. On **htc-login**, start an interactive GPU shell and bootstrap:

   ```bash
   srun -p interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
   export PE_HUB_ROOT=$DATA/pe-hub
   bash $PE_HUB_ROOT/scripts/cluster/oxford-arc/setup_interactive.sh
   ```
3. Configure local paths:

   ```bash
   cp $PE_HUB_ROOT/scripts/cluster/oxford-arc/env.sh.example \
      $PE_HUB_ROOT/scripts/cluster/oxford-arc/env.sh
   # edit PE_HUB_ROOT, ARC_MODULES, CONDA_ENV (\$DATA prefix), ARC_MAIL_USER,
   # optional ARC_GPU_CONSTRAINT. Confirm module name: module spider Anaconda
   ```

   Jobs load `ARC_MODULES` then always `conda activate $CONDA_ENV` via
   `job_env.sh` (even if you submitted from an already-activated shell).
   `--export=ALL` plus `module load Anaconda3` would otherwise leave the
   module `python` first on `PATH`, and datasheet-benchmark would fail with
   `No module named 'pe_common'`. Do **not** rely on `conda init` in `.bashrc`
   for batch jobs.

## Sync artifacts (laptop to ARC)

Run these **on the laptop** (ARC cannot SSH back into WSL). They rsync every
DVC-tracked folder (paths from repo `*.dvc` files), gitignored `env.sh`, and
vendor-weight evaluation results
(`scripts/experiments/base-model-eval/results/`) in **one** rsync, so you type
the SSH password once. No git commits.

`ARC_PROJECT` is the share name under `/data/`. It is taken from the
environment or from `env.sh`. If it is still unset, the script prompts, the
same way it prompts for the ARC username. `LIST=1` skips both prompts.

```bash
# VPN, or ProxyJump gateway.arc.ox.ac.uk for htc-login in ~/.ssh/config
./scripts/cluster/oxford-arc/pull_from_arc.sh            # prompts for username
./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER   # ARC → laptop
./scripts/cluster/oxford-arc/push_to_arc.sh YOUR_ARC_USER     # laptop → ARC

DRY_RUN=1 ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
LIST=1 ./scripts/cluster/oxford-arc/pull_from_arc.sh     # show paths, no SSH
SKIP=datasets/reference ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
ONLY=env ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
ONLY=results,slurm_output ./scripts/cluster/oxford-arc/push_to_arc.sh YOUR_ARC_USER
ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
ONLY=vendor-eval ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
EXTRA=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
DELETE=1 ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER   # dest extras removed
```

`ONLY=pridict2-repro` pulls the PRIDICT2 reproduction weight dirs, `local_registry.json`,
and `scripts/experiments/pridict2-reproduction/state/` (see
`scripts/experiments/pridict2-reproduction/supplementary_artifacts.txt`). Then zip for
supplementary files with
`./scripts/experiments/pridict2-reproduction/pack_supplementary_weights.sh`.

`ONLY=scratch-weights` pulls the preferred scratch-benchmark weight dirs (see
[`scripts/experiments/scratch-benchmark/README.md`](../../experiments/scratch-benchmark/README.md#preferred-trained-weights-pe-ensemble)).

`ONLY=vendor-eval` syncs only `scripts/experiments/base-model-eval/results/`.
That directory is also in the default push and pull set.

`pull_env_from_arc.sh` still exists as `ONLY=env` (env.sh only).

Off-campus: put `ProxyJump gateway.arc.ox.ac.uk` for `htc-login` in `~/.ssh/config`.

Do not rsync the whole repo. Code stays on `git pull`. Regenerable caches
(`datasets/exported/`, `standardized/`, `formatted/`, conda envs) stay on each
machine.

## DVC (optional pointers)

`datasets/raw/` and vendor/plugin blobs under `services/pe-ensemble/weights/`
stay in **git**. Existing `*.dvc` files are the rsync path list. You do **not**
need `dvc add` / `dvc push` / pointer commits to copy those folders — use
[rsync](#sync-artifacts-laptop-to-arc).

| In git already | In rsync set today | Usually skip |
|----------------|--------------------|--------------|
| `datasets/raw/` | — | — |
| Vendor / plugin weights | — | — |
| `datasets/reference/` | yes (`datasets/reference.dvc`) | — |
| `/results`, `/slurm_output` | yes (root `*.dvc`) | — |
| `scripts/experiments/scratch-benchmark/results/` | yes | — |
| `scripts/experiments/base-model-eval/results/` | yes (vendor weight eval) | — |
| Trained weights `*__*__*__*/` | add a `.dvc` (or rsync that path yourself) | smoke, failed trials |
| `training_presets_local/*.yaml` | same | will re-tune anyway |
| `tuning_studies/*.db` | same | presets + weights are enough |
| Standardized/formatted caches | — | regenerate with `pedb init` |

To add another folder to the rsync set, `dvc add` it once so a `*.dvc` pointer
exists (you can leave the DVC store unused). Or pass `ONLY=` / extra rsync.

## Submit jobs

From **htc-login** (scheduler only — do not run peen on the login node):

```bash
cd $DATA/pe-hub
source scripts/cluster/oxford-arc/env.sh   # optional; submit.sh sources it

# Validate when the job would start
DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh

# Full HPO (default: short, 12h, 1 GPU) — Optuna study resumes if re-submitted
./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh

# After tuning: train + register weights
./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh

# Smoke on short + any GPU
SMOKE=1 ARC_PARTITION=short ARC_TIME=01:00:00 \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
```

Other pridict2-reproduction stages use the same pattern (`02_…`, `04_…`, `05_…`, …).

**Full reproduction with SLURM dependencies** (`afterok` chain: tune ∥ train → FT → ensemble):

```bash
./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
# ARC_DEPENDENCY=afterok:JOBID ./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh
```

**Scratch benchmark** (DeepPrime / OPED / PRIDICT2 × 7 datasets × 3 seeds; 10 Optuna trials + final train + eval per short L40S job):

```bash
./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh   # 63 jobs
RUN_ID=<id> ./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh
```

See [`scripts/experiments/scratch-benchmark/README.md`](../../experiments/scratch-benchmark/README.md).

Pin L40S (default in env.sh) or A100:

```bash
ARC_GPU_CONSTRAINT='gpu_sku:L40S' \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
# or:
./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh -- --constraint='gpu_sku:A100'
```

## Monitor

```bash
squeue --clusters=htc -u $USER
squeue --clusters=all -u $USER
scancel --clusters=htc <jobid>
tail -f $PE_HUB_ROOT/slurm-<jobid>.out
```

## Artifacts

Most outputs live under the repo on `$DATA` and are gitignored. Copy them with
[rsync](#sync-artifacts-laptop-to-arc); do not commit `.dvc` pointers for routine
laptop ↔ ARC sync.

| What | Where |
|------|--------|
| Reference genomes | `datasets/reference/` (rsync via `datasets/reference.dvc`) |
| HPO presets | `services/pe-ensemble/config/training_presets_local/` |
| Trained weights | `services/pe-ensemble/weights/*__*__*__*/` |
| Benchmark results | `scripts/experiments/scratch-benchmark/results/<RUN_ID>/` |
| Vendor weight eval | `scripts/experiments/base-model-eval/results/<RUN_ID>/` |
| Slurm logs / eval dumps | `slurm_output/`, `results/` |
| Shipped defaults | `services/pe-ensemble/config/training_presets/` (git) |

Override roots with `WEIGHTS_ROOT`, `TUNING_STUDIES_ROOT`, `TRAINING_PRESETS_ROOT`.

## Bring results home

On the laptop:

```bash
./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
# env.sh only:
./scripts/cluster/oxford-arc/pull_env_from_arc.sh YOUR_ARC_USER
```

That covers current DVC-tracked trees (`datasets/reference/`, `results/`,
`slurm_output/`, scratch-benchmark `results/`), vendor-weight evaluation
results (`scripts/experiments/base-model-eval/results/`), and cluster `env.sh`.
For the
PRIDICT2 reproduction trained weights:

```bash
ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh YOUR_ARC_USER
./scripts/experiments/pridict2-reproduction/pack_supplementary_weights.sh
```

For other trained weights or `training_presets_local/` that are not yet in a
`*.dvc` file, rsync those paths once, pass `EXTRA=…`, or add a pointer so they
join the default set.

Only if you deliberately publish a shared baseline:

```bash
MODEL=pridict2 ./scripts/cluster/oxford-arc/promote_presets.sh --apply
git add services/pe-ensemble/config/training_presets/
git commit -m "Promote curated training presets"
git push
```

## Walltime guidance

| Job                                        | Suggested partition / time                              |
| ------------------------------------------ | ------------------------------------------------------- |
| `SMOKE=1` tune                           | `short` / 1h                                          |
| Scratch-benchmark 01 (one seed: 10 trials + train + eval) | `short` / 12h L40S; re-queue same `INDEX` if ClinVar times out |
| Scratch-benchmark 01 (63 jobs)                    | `submit.sh 01_tune_matrix.sh` (default per-seed fan-out) |
| Packed cell (3 seeds in one job)                  | `SUBMIT_SEEDS=0` + `medium` / 48h              |
| Single train / fine-tune                           | `short`–`medium`                               |
| Multi-day HPO                                      | `long` with explicit `--time` (e.g. `7-00:00:00`) |

Each scratch-benchmark GPU job is **one seed**: 10 Optuna trials plus `register_best_weights` (11 trains) and eval. There is no separate train stage. After a 12h kill, re-submit the same `RUN_ID` + `INDEX` with `SKIP_IF_DONE=1`: Optuna resumes from `TUNING_STUDIES_ROOT` (orphan `RUNNING` trials are marked `FAIL`), and a finished HPO checkpoint (`hpo_done`) skips straight to final train/eval. Keep the shared `$DATA` checkout’s `tuning_studies/*.db` intact; a search-space fingerprint change starts a new study.

## Checklist before first real submit

- [ ] Account can reach `htc-login` (VPN or gateway)
- [ ] Checkout under `$DATA`; Anaconda module + env prefix under `$DATA/envs/`
- [ ] `peen devices` shows CUDA on an interactive GPU allocation
- [ ] `datasets/` prepared (`pedb init`; `pull_from_arc.sh` for genomes)
- [ ] **Local smoke passed** (mini data, 1 trial): `SMOKE=1 ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh`
- [ ] **Local preflight** (optional full pipeline on mini data): `./scripts/cluster/oxford-arc/preflight.sh`
- [ ] `env.sh` points at the checkout
- [ ] `DRY_RUN=1` submit succeeds
- [ ] Optional: `ARC_MAIL_USER` for END/FAIL mail

### Local smoke (before ARC — run this first)

`SMOKE=1` on any reproduction stage script: subsampled `DATA_ROOT` in `/tmp`, 1 Optuna trial, 2-fold CV, isolated weights/presets. Exercises the same convert → train path as ARC.

```bash
conda activate pe-hub
SMOKE=1 ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh
# or equivalent:
SMOKE=1 ./scripts/cluster/oxford-arc/preflight.sh
DEVICE=cuda:0 KEEP_WORK=1 SMOKE=1 ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh
```

On ARC, pass `SMOKE=1` via `submit.sh` (uses full data by default; add `SMOKE_FULL_DATA=1` explicitly if you already set mini data locally).

### Local preflight (optional, full pipeline)

Runs tune + train + merge + fine-tune + evaluate + ensemble on mini data (omit `SMOKE=1` for the full preflight):

```bash
./scripts/cluster/oxford-arc/preflight.sh
DEVICE=cuda:0 KEEP_WORK=1 ./scripts/cluster/oxford-arc/preflight.sh
```

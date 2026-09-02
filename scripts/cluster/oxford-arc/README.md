# Oxford ARC — submit pe-hub tuning / training

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

   Build datasets on ARC with `pedb init`. Pull shared reference genomes with
   `dvc pull` — see [DVC on ARC](#dvc-selective-artifacts).
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

   Jobs load `ARC_MODULES` then `conda activate $CONDA_ENV` via `job_env.sh`.
   Do **not** rely on `conda init` in `.bashrc` for batch jobs.

## DVC (selective artifacts)

`datasets/raw/` and vendor/plugin blobs under `services/pe-ensemble/weights/`
stay in **git**. Use DVC only for bulky artifacts you deliberately choose to
keep — not a bulk sync of every run.

| In git already | Worth `dvc add` when… | Usually skip |
|----------------|----------------------|--------------|
| `datasets/raw/` | — | — |
| Vendor / plugin weights | — | — |
| `datasets/reference/` | always (already tracked: `datasets/reference.dvc`) | — |
| Trained weights `*__*__*__*/` | final/best run you want to evaluate or compare | smoke, failed, superseded trials |
| `training_presets_local/*.yaml` | HPO finished; want reproducible training | will re-tune anyway |
| Benchmark results | run you care about (summary CSV, eval JSONL) | intermediate matrix cells |
| `tuning_studies/*.db` | resuming Optuna on another machine | presets + weights are enough |
| `scripts/experiments/*/state/` | — | tiny ID files; recreate from logs |
| Standardized/formatted caches | — | regenerate with `pedb init` |

Store path: **`/data/<ARC_PROJECT>/<USER>/dvc-store`** (same layout as
`$DATA/pe-hub`). Run `setup_dvc_remote.sh` once per machine so gitignored
`.dvc/config.local` points at your store (local path on ARC, SSH from laptop).

```bash
conda activate pe-hub
pip install 'dvc[ssh]'   # once; setup_interactive.sh does this on ARC
ARC_USER=you ./scripts/cluster/oxford-arc/setup_dvc_remote.sh

# Shared reference genomes (already in repo):
dvc pull datasets/reference.dvc

# After a run you want to keep (example — one weight set):
dvc add services/pe-ensemble/weights/pridict2/pridict2__custom__20260901__abc123
dvc push
git add services/pe-ensemble/weights/pridict2/pridict2__custom__20260901__abc123.dvc
git commit -m 'Track pridict2 benchmark weights'
git push

# On the other machine:
git pull && dvc pull
```

Off-campus SSH: put `ProxyJump gateway.arc.ox.ac.uk` for `htc-login` in
`~/.ssh/config` (key auth; DVC’s SSH client is picky about password/2FA jumps).

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

**Scratch benchmark** (DeepPrime / OPED / PRIDICT2 × library1 / library-diverse / ClinVar, holdout_3 HPO):

```bash
./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh   # scratch-benchmark/
./scripts/cluster/oxford-arc/submit.sh 02_train_matrix.sh
./scripts/cluster/oxford-arc/submit.sh 03_evaluate_matrix.sh
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

Most outputs live under the repo on `$DATA` and are gitignored. Only version
what you would regret losing — see [DVC table](#dvc-selective-artifacts) above.

| What | Where |
|------|--------|
| Reference genomes | `datasets/reference/` (`datasets/reference.dvc`) |
| HPO presets | `services/pe-ensemble/config/training_presets_local/` |
| Trained weights | `services/pe-ensemble/weights/*__*__*__*/` |
| Benchmark results | `scripts/experiments/scratch-benchmark/results/<RUN_ID>/` |
| Shipped defaults | `services/pe-ensemble/config/training_presets/` (git) |

Override roots with `WEIGHTS_ROOT`, `TUNING_STUDIES_ROOT`, `TRAINING_PRESETS_ROOT`.

## Bring results home

Pick what to keep, then `dvc add` + `dvc push` on ARC and commit the `.dvc`
pointers. On laptop: `git pull && dvc pull`.

```bash
# example: one weight set + its preset YAML
dvc add services/pe-ensemble/weights/deepprime/deepprime__custom__20260901__abc123
dvc add services/pe-ensemble/config/training_presets_local/deepprime.yaml
dvc push
git add '*.dvc' && git commit -m 'deepprime library1 benchmark' && git push
```

Pull the gitignored cluster `env.sh` (partition, paths, modules) from ARC:

```bash
./scripts/cluster/oxford-arc/pull_env_from_arc.sh          # prompts for username
./scripts/cluster/oxford-arc/pull_env_from_arc.sh wolf6973
```

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
| Full HPO (`N_TRIALS=20`, `CV_FOLDS=5`) | `medium` 48h; re-submit to resume Optuna if needed    |
| Single train / fine-tune                   | `short`–`medium`                                   |
| Multi-day HPO                              | `long` with explicit `--time` (e.g. `7-00:00:00`) |

Default peen HPO is ~100 fold trains (20×5) plus a final register — plan for multi-hour / multi-day, not a login-node run.

## Checklist before first real submit

- [ ] Account can reach `htc-login` (VPN or gateway)
- [ ] Checkout under `$DATA`; Anaconda module + env prefix under `$DATA/envs/`
- [ ] `peen devices` shows CUDA on an interactive GPU allocation
- [ ] `datasets/` prepared (`pedb init`; `dvc pull datasets/reference.dvc` for genomes)
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

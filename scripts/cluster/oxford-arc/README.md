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

1. Clone or rsync pe-hub under `$DATA` (from your laptop, not the login node for big trees):

   ```bash
   # Prefer the helper (skips weights/jobs/eval artifacts + exported/formatted caches):
   ./scripts/cluster/oxford-arc/sync_to_arc.sh
   DRY_RUN=1 ./scripts/cluster/oxford-arc/sync_to_arc.sh   # preview
   ```

   Manual equivalent (VPN → htc-login; use gateway off-net):

   ```bash
   rsync -avz --progress \
     --exclude '.git/objects/' --exclude '__pycache__/' \
     --exclude 'datasets/exported/' --exclude 'datasets/formatted/' --exclude 'datasets/catalog/' \
     --exclude 'datasets/reference/' --exclude 'results/' --exclude '.dvc/cache/' \
     --exclude 'services/pe-ensemble/weights/' \
     --exclude 'services/pe-ensemble/tuning_studies/' \
     --exclude 'services/pe-ensemble/config/training_presets_local/' \
     --exclude 'services/pe-ensemble/jobs/' \
     --exclude 'services/pe-ensemble/eval_jobs/' \
     --exclude 'services/pe-ensemble/ensemble_jobs/' \
     --exclude 'services/pe-ensemble/validation_jobs/' \
     --exclude 'services/pe-ensemble/checkpoints/' \
     --exclude 'artifacts/' --exclude 'checkpoints/' \
     --exclude 'scripts/experiments/pridict2-reproduction/state/' \
     ./ you@htc-login.arc.ox.ac.uk:/data/<project>/<user>/pe-hub/
   ```

   Prefer transferring into **`$DATA`**. Gateway does not expose `$HOME`.
   `datasets/raw` + `datasets/standardized` are kept by default; add
   `SKIP_STANDARDIZED=1` on `sync_to_arc.sh` to rebuild standardized on ARC.
   Reference genomes and eval `results/` are **DVC** (not rsynced); see
   [DVC on ARC](#dvc-reference-genomes-and-eval-results) below.
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

## DVC (reference genomes and eval results)

`datasets/raw/` and vendor/plugin blobs under `services/pe-ensemble/weights/`
stay in **git** so clones share them. DVC is only for bulky machine-local
artifacts:

| Tracked | Git | DVC (ARC store) |
|---------|-----|-----------------|
| Study raw files | `datasets/raw/` | — |
| Vendor / plugin weights | `services/pe-ensemble/weights/` (not `*__*__*__*` IDs) | — |
| hg38 / mm39 FASTA | — | `datasets/reference.dvc` |
| Eval outputs | — | `results.dvc` |
| A trained weight set you want versioned | — | `dvc add services/pe-ensemble/weights/<model>/<id>` |

Store path is **`/data/<ARC_PROJECT>/<USER>/dvc-store`** (project is shared by
the group; each member keeps their own store, same layout as `$DATA/pe-hub`).
`.dvc/config` only has a `USER` placeholder — always run the setup script so
gitignored `.dvc/config.local` has your real path (USB-style mount names never
belong in git).

```bash
conda activate pedb
pip install 'dvc[ssh]'   # once
# laptop (VPN) or ARC — writes /data/coml-deepcmb/$ARC_USER/dvc-store:
ARC_USER=you ./scripts/cluster/oxford-arc/setup_dvc_remote.sh
# first machine that has the data:
dvc push
# the other machine (same ARC_USER / same store):
dvc pull
```

Off-campus SSH: put `ProxyJump gateway.arc.ox.ac.uk` for `htc-login` in
`~/.ssh/config` (key auth; DVC’s SSH client is picky about password/2FA jumps).

HPO presets, Optuna DBs, and live trained checkpoints still use
`sync_from_arc.sh` / rsync.

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

## Artifacts (persist under the repo on `$DATA`)

| What | Where | How to bring home |
|------|--------|-------------------|
| Reference genomes | `datasets/reference/` | **DVC** (`dvc pull`) |
| Eval run outputs | `results/` | **DVC** (`dvc pull`) |
| HPO dataset presets | `config/training_presets_local/` | **rsync** (gitignored; empty until you tune or sync) |
| Shipped model defaults | `config/training_presets/` | **Git** (defaults; promote rarely) |
| Trained weights + index | `weights/*__*__*__*/` + `local_registry.json` | **rsync** (gitignored) |
| Pipeline state IDs | `scripts/experiments/pridict2-reproduction/state/` | rsync (optional) |
| Optuna study DB | `tuning_studies/*.db` | rsync only if resuming elsewhere |

Paths above are under `services/pe-ensemble/` except pipeline state. Override with `WEIGHTS_ROOT` / `TUNING_STUDIES_ROOT` / `TRAINING_PRESETS_ROOT` (local overlay).

## Bring results home

Routine tuning does **not** go to GitHub. Sync local presets + weights:

```bash
# on laptop
./scripts/cluster/oxford-arc/sync_from_arc.sh
peen weights --model pridict2
./scripts/hyperparameter/check_tuning_status.sh pridict2
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
- [ ] `datasets/` prepared (`pedb init` or rsynced standardized/formatted caches)
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

# Oxford ARC — submit pe-hub tuning / training

# Official docs: [ARC User Guide](https://arc-user-guide.readthedocs.io/en/latest/)

# GPUs live only on the **htc** cluster.

## What to use

| Need                     | Choice                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Cluster                  | `htc` (`htc-login.arc.ox.ac.uk`)                                                                           |
| Login (on Uni net / VPN) | `ssh you@htc-login.arc.ox.ac.uk`                                                                             |
| Off-net                  | `ssh you@gateway.arc.ox.ac.uk` then hop to `htc-login`                                                     |
| Repo + data              | `$DATA/...` (project share, ~5 TiB). Avoid `$HOME` (15 GiB) for conda/datasets                             |
| Partitions               | `short` ≤12h · `medium` ≤48h · `long` ≤30d (default 1d unless `--time` set) · `devel` 10m test |
| GPU                      | `#SBATCH --gres=gpu:1` (+ optional `--constraint='gpu_sku:A100'`)                                          |
| Build / pip / conda      | **interactive** node, not login: `srun -p interactive --gres=gpu:1 --pty bash`                         |

Co-investment GPU nodes are often limited to **short** (12h). Prefer ARC-owned L40S/A100 for **medium**/**long** HPO.

## One-time setup

1. Clone or rsync pe-hub under `$DATA` (from your laptop, not the login node for big trees):

   ```bash
   # On ARC: mkdir -p $DATA && pwd   # note full /data/<project>/<user> path
   # Locally (via gateway if off-net):
   rsync -avz --exclude .git/objects --exclude __pycache__ \
     ./pe-hub/ you@gateway.arc.ox.ac.uk:/data/<project>/<user>/pe-hub/
   ```

   Prefer transferring into **`$DATA`**. Gateway does not expose `$HOME`.
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
   # edit PE_HUB_ROOT, CONDA_ROOT, ARC_MAIL_USER, optional ARC_GPU_CONSTRAINT
   ```

## Submit jobs

From **htc-login** (scheduler only — do not run peen on the login node):

```bash
cd $DATA/pe-hub
source scripts/cluster/oxford-arc/env.sh   # optional; submit.sh sources it

# Validate when the job would start
DRY_RUN=1 ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh

# Full HPO (default: medium, 48h, 1 GPU) — Optuna study resumes if re-submitted
./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh

# After tuning: train + register weights
./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh

# Smoke on short + any GPU
SMOKE=1 ARC_PARTITION=short ARC_TIME=01:00:00 \
  ./scripts/cluster/oxford-arc/submit.sh 01_tune_base_library1.sh
```

Other pridict2-reproduction stages use the same pattern (`02_…`, `04_…`, `05_…`, …).

Pin a GPU type:

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
ARC_REMOTE=you@gateway.arc.ox.ac.uk:/data/<project>/<user>/pe-hub \
  ./scripts/cluster/oxford-arc/sync_from_arc.sh
peen weights --model pridict2
./scripts/hyperparameter/check_tuning_status.sh pridict2
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
- [ ] Checkout + conda under `$DATA`
- [ ] `peen devices` shows CUDA on an interactive GPU allocation
- [ ] `datasets/` prepared (`pedb init` or rsynced standardized/formatted caches)
- [ ] **Local preflight passed**: `./scripts/cluster/oxford-arc/preflight.sh`
- [ ] `env.sh` points at the checkout
- [ ] `DRY_RUN=1` submit succeeds
- [ ] Optional: `ARC_MAIL_USER` for END/FAIL mail

### Local preflight (before ARC)

Runs a tiny sampled DATA_ROOT through the same peen shapes as the reproduction
pipeline (single-sheet HPO, merge+author-fold HPO, train, fine-tune, evaluate,
ensemble) with isolated presets/weights:

```bash
./scripts/cluster/oxford-arc/preflight.sh
DEVICE=cuda:0 KEEP_WORK=1 ./scripts/cluster/oxford-arc/preflight.sh
```

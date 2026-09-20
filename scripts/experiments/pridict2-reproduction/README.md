# PRIDICT 2.0 transfer + ensemble reproduction
#
# Reproduces the Mathis et al. recipe via PE-DB + PE Ensemble:
#
# 1. Base train on **PRIDICT library1**
# 2. Base train on **library1 + DeepPrime ClinVar** (DeepPrime `original_fold`
#    propagated onto overlapping library1 loci by `target_uid`)
# 3. Fine-tune each base on **library-diverse** with vendor-style **run_x**:
#    one `holdout_3` model per author fold (0–4) × cell (hek / k562) — fold `x`
#    held out as test; **no** internal CV + final export
# 4. **Mean-ensemble** Model A_x + Model B_x per cell × fold; evaluate on fold `x`
#
# Library1 uses a random **holdout_3** (70/15/15) for *this* reproduction (author
# PRIDICT1 folds were never published). Stage 01 HPO and stage 03 final train
# share that protocol. Vendor PRIDICT2/OptiPrime training used the full
# library1 sheet — every locus is training data for those checkpoints.
# L1+ClinVar base (02/04) also uses **holdout_3** (one train/val/test per trial,
# not 5-fold CV), with DeepPrime `original_fold == -1` held out as test via
# `--use-original-fold`.
#
# **Loss:** PRIDICT2 is trained with a single edit-efficiency head (`MSEloss` on
# `averageedited`, mapped from `editing_efficiency` in standardized data). All
# reproduction stages use MSEloss; KLD/CE distribution training is not used.
#
## Quick start
#
# Local smoke (mini data, 1 trial — run before ARC):
#
# ```bash
# conda activate <env> && ./scripts/install-clis.sh
# SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/pridict2-reproduction/01_tune_base_library1.sh
# # or: SMOKE=1 ./scripts/cluster/oxford-arc/preflight.sh
# ```
#
# Full pipeline smoke (all stages, mini data):
#
# ```bash
# SMOKE=1 DEVICE=mps ./scripts/experiments/pridict2-reproduction/run_all.sh
# ```
#
# Full run (skips stages that already have presets / state by default):
#
# ```bash
# DEVICE=cuda:0 SKIP_IF_TUNED=1 SKIP_IF_DONE=1 \
#   ./scripts/experiments/pridict2-reproduction/run_all.sh
# ```
#
# Resume / subset:
#
# ```bash
# ONLY=06,07 ./scripts/experiments/pridict2-reproduction/run_all.sh
# SKIP=01,02 ./scripts/experiments/pridict2-reproduction/run_all.sh
# CELL_LINE=hek FOLD=0 ./scripts/experiments/pridict2-reproduction/05_tune_finetune_library_diverse.sh
# ```
#
## Scripts
#
# | Script | Role |
# |--------|------|
# | `run_all.sh` | Orchestrator (tune → train → fine-tune → ensemble) |
# | `01_tune_base_library1.sh` | HPO base on library1 (holdout_3) |
# | `02_tune_base_l1_clinvar.sh` | HPO base on L1+ClinVar (holdout_3 + ClinVar fold −1 test) |
# | `03_train_base_library1.sh` | Train + register library1 base weights |
# | `04_train_base_l1_clinvar.sh` | Train + register L1+ClinVar base (holdout_3) |
# | `05_tune_finetune_library_diverse.sh` | Model A FT: library1 base → library-diverse (`CELL_LINE=` / `FOLD=`) |
# | `06_finetune_transfer.sh` | Model B FT: L1+ClinVar base → library-diverse (`CELL_LINE=` / `FOLD=`) |
# | `07_ensemble_by_cell_line.sh` | Mean ensemble A_x + B_x per cell × fold; eval on fold x |
# | `_common.sh` | Shared env, state helpers |
# | `submit_arc_pipeline.sh` | ARC: submit 01–07; 05/06/07 = one short job per cell × fold |
#
## State
#
# Weights IDs and logs are written under `state/` (gitignored):
#
# - `base_library1`, `base_l1_clinvar`
# - `ft_base_library1_{hek,k562}_fold{0..4}`
# - `ft_base_l1_clinvar_{hek,k562}_fold{0..4}`
# - `ensemble_{hek,k562}_fold{0..4}`
#
# Override with `STATE_DIR=/path/to/dir`.
#
## Oxford ARC (cluster)
#
# GPU jobs go on **htc**. Submit wrappers + setup notes:
# [`../../cluster/oxford-arc/`](../../cluster/oxford-arc/README.md).
#
# Full pipeline with SLURM dependencies. Stages **05**, **06**, and **07** each
# submit **one short job per (cell × fold)** (10 each). Defaults: FT **3h**,
# ensemble **1h**.
#
# ```bash
# source scripts/cluster/oxford-arc/env.sh
# ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
# # or: SKIP=01,02 ./.../submit_arc_pipeline.sh   # presets already done
# #      ONLY=05,06,07 …                           # fold FT + ensemble only
# ```
#
# Single stage (optional dependency):
#
# ```bash
# ARC_DEPENDENCY=afterok:123456 ./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh
# CELL_LINE=hek FOLD=0 ./scripts/cluster/oxford-arc/submit.sh 05_tune_finetune_library_diverse.sh
# ```
#
## Pull trained weights (laptop) + GitHub Release (transparency)
#
# Weight IDs: `weights_id_map.tsv`. Paths: `supplementary_artifacts.txt`
# (2 bases + 20 fine-tunes). Blobs are **not** git-tracked — ship on the shared
# `experiment-weights-v1` release (with scratch-benchmark weights) for
# transparency; no local registry install.
#
# ```bash
# ONLY=pridict2-repro ./scripts/cluster/oxford-arc/pull_from_arc.sh "$USER"
# ./scripts/experiments/pridict2-reproduction/pack_supplementary_weights.sh
# # → txt/supplementary/pridict2-reproduction-weights.zip (~114 MB, gitignored)
#
# # Combined release (both zips); create once from repo root:
# gh release create experiment-weights-v1 \
#   txt/supplementary/scratch-benchmark-weights.zip \
#   txt/supplementary/pridict2-reproduction-weights.zip \
#   --title "Experiment weights (scratch + PRIDICT2 repro)" \
#   --notes "See README Installation → Experiment weights."
# ```
#
# Download only:
# ```bash
# gh release download experiment-weights-v1 -p pridict2-reproduction-weights.zip -D /tmp
# ```
#
## Shared HPO helpers
#
# Optuna helpers (base stages 01/02) live in
# [`../../hyperparameter/`](../../hyperparameter/README.md).
# Fine-tune stages 05/06 do **not** run Optuna; they use peen merge / baseline HPs.

# PRIDICT 2.0 transfer + ensemble reproduction
#
# Reproduces the Mathis et al. recipe via PE-DB + PE Ensemble:
#
# 1. Base train on **PRIDICT library1**
# 2. Base train on **library1 + DeepPrime ClinVar** (DeepPrime `original_fold`
#    propagated onto overlapping library1 loci by `target_uid`)
# 3. Fine-tune both bases on **library-diverse** HEK and K562 → four models
# 4. **Mean-ensemble** the two fine-tunes per cell line
#
# Library1 uses a random **holdout_3** (70/15/15) for *this* reproduction (author
# PRIDICT1 folds were never published). Stage 01 HPO and stage 03 final train
# share that protocol. Vendor PRIDICT2/OptiPrime training used the full
# library1 sheet — every locus is training data for those checkpoints.
# L1+ClinVar base (02/04) also uses **holdout_3** (one train/val/test per trial,
# not 5-fold CV), with DeepPrime `original_fold == -1` held out as test via
# `--use-original-fold`. Library-diverse fine-tune (05/06) uses author **5-fold
# CV** (`testset_fold` 0–4) with **no outer random holdout**, then evaluates on
# fold `LD_TEST_FOLD` (default 4).
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
# SKIP=01,02,05 ./scripts/experiments/pridict2-reproduction/run_all.sh
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
# | `05_tune_finetune_library_diverse.sh` | HPO fine-tune (HEK + K562): author CV5, then eval fold 4 |
# | `06_finetune_transfer.sh` | Four transfer fine-tunes (author CV5) + eval fold 4 |
# | `07_ensemble_by_cell_line.sh` | Mean ensemble per cell line (eval on fold 4) |
# | `_common.sh` | Shared env, state helpers |
# | `submit_arc_pipeline.sh` | ARC: submit 01–07 with SLURM `afterok` dependencies |
#
## State
#
# Weights IDs and logs are written under `state/` (gitignored):
#
# - `base_library1`, `base_l1_clinvar`
# - `ft_base_library1_{hek,k562}`, `ft_base_l1_clinvar_{hek,k562}`
# - `ensemble_{hek,k562}`
#
# Override with `STATE_DIR=/path/to/dir`.
#
## Oxford ARC (cluster)
#
# GPU jobs go on **htc**. Submit wrappers + setup notes:
# [`../../cluster/oxford-arc/`](../../cluster/oxford-arc/README.md).
#
# Full pipeline with SLURM dependencies (tune ∥ train → fine-tune → ensemble):
#
# ```bash
# source scripts/cluster/oxford-arc/env.sh
# ./scripts/experiments/pridict2-reproduction/submit_arc_pipeline.sh
# # or: SKIP=01,02 ./.../submit_arc_pipeline.sh   # presets already done
# ```
#
# Single stage (optional dependency):
#
# ```bash
# ARC_DEPENDENCY=afterok:123456 ./scripts/cluster/oxford-arc/submit.sh 03_train_base_library1.sh
# ```
#
## Shared HPO helpers
#
# Optuna helpers live in [`../../hyperparameter/`](../../hyperparameter/README.md).

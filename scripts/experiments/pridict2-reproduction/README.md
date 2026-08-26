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
# Library1 uses a random holdout (author PRIDICT1 folds were unavailable).
# Other PRIDICT2 sheets use random CV + outer test for HPO (not author
# `testset_fold`), except the merged L1+ClinVar base which aligns to DeepPrime folds.
#
# **Loss:** only `pridict1/library1` has the full outcome trio
# (`averageedited`/`averageunedited`/`averageindel`). peen refuses KL/CE when
# any trio column is missing or NaN. Merge (L1+ClinVar) and library-diverse FT
# stages force `loss_func=MSEloss` (intended-edit only). Library1 base HPO/train
# keeps the default `KLDloss`.
#
## Quick start
#
# ```bash
# conda activate <env> && ./scripts/install-clis.sh
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
# | `01_tune_base_library1.sh` | HPO base on library1 |
# | `02_tune_base_l1_clinvar.sh` | HPO base on L1+ClinVar (DeepPrime folds) |
# | `03_train_base_library1.sh` | Train + register library1 base weights |
# | `04_train_base_l1_clinvar.sh` | Train + register L1+ClinVar base weights |
# | `05_tune_finetune_library_diverse.sh` | HPO fine-tune stage (HEK + K562) |
# | `06_finetune_transfer.sh` | Four transfer fine-tunes from the two bases |
# | `07_ensemble_by_cell_line.sh` | Mean ensemble per cell line |
# | `_common.sh` | Shared env, state helpers |
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
## Shared HPO helpers
#
# Optuna helpers live in [`../../hyperparameter/`](../../hyperparameter/README.md).

# Hyperparameter helpers (shared by CLI experiments)
#
# | Script | Purpose |
# |--------|---------|
# | `check_tuning_status.sh` | List / check whether a model×dataset Optuna preset exists |
# | `tune_hpo_cv5.sh` | Generic `peen tune` with 5-fold CV + outer test holdout |
# | `_common.sh` | Shared env defaults (`N_TRIALS`, `DEVICE`, `SKIP_IF_TUNED`, …) |
#
# Dataset-specific experiment recipes live under [`../experiments/`](../experiments/README.md).
#
# ```bash
# ./scripts/hyperparameter/check_tuning_status.sh pridict2
# ./scripts/hyperparameter/check_tuning_status.sh pridict2 minsepie/library_insert_set12/hek293t/pe2
#
# ./scripts/hyperparameter/tune_hpo_cv5.sh --model pridict2 \
#   --dataset-name demo --study minsepie --dataset library-insert-set12 \
#   --cell-line hek293t --pe-system pe2
#
# SKIP_IF_TUNED=1 ./scripts/experiments/tune_pridict2_minsepie.sh
# ```
#
# Artifacts:
# - Local HPO presets: `services/pe-ensemble/config/training_presets_local/<model>.yaml` (gitignored)
# - Shipped defaults: `services/pe-ensemble/config/training_presets/<model>.yaml`
# - Optuna DBs: `services/pe-ensemble/tuning_studies/*.db`

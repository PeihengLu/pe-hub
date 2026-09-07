# Hyperparameter search

Generic Optuna runners over `peen tune`. Dataset-specific recipes that call these
live under [`../experiments/`](../experiments/README.md).

| Script | Purpose |
|--------|---------|
| `tune_hpo_cv5.sh` | `peen tune` with 5-fold CV plus an outer test holdout |
| `tune_hpo_holdout3.sh` | `peen tune` with `holdout_3` (70/15/15) |
| `check_tuning_status.sh` | Report whether a model × dataset preset already exists |
| `_common.sh` | Shared defaults, option handling, skip logic — sourced by all of the above |

## How a runner is put together

Each runner is thin: it sets its protocol defaults, sources `_common.sh`, builds
a `peen tune` argument array, and executes it. So when a flag's value is
unclear, the answer is in `_common.sh` rather than the runner.

`_common.sh` provides:

- **Env-overridable defaults** — `N_TRIALS`, `DEVICE`, `CV_FOLDS`, `TEST_PCT`,
  `SPLIT_RANDOM_STATE`, `SMOKE`.
- **`SMOKE=1`** — shrinks trials/folds for a fast end-to-end check.
- **`require_peen`** — fails early with the install command if `peen` is absent.
- **`maybe_skip_if_tuned`** — with `SKIP_IF_TUNED=1`, exits successfully when a
  preset for that model × dataset already exists. This makes experiment scripts
  idempotent and safe to re-run after a partial sweep.
- **`append_register_best_weights`** — adds `--register-best-weights` only when
  enabled.

## Default protocol

- Outer test holdout (`--test-pct`, default 0.15)
- 5-fold CV on the remainder drives the Optuna objective
- DeepPrime exception: `--use-original-fold` uses the author folds (`-1` = test)
- Merged DeepPrime ClinVar + PRIDICT library1 uses `--merge --use-original-fold`,
  where overlapping library1 loci inherit the DeepPrime `original_fold`

## Weight registration differs by protocol

`tune_hpo_cv5.sh` registers a weight set for the best trial; `tune_hpo_holdout3.sh`
does not. The CV objective is a k-fold mean, so its best trial is worth keeping,
whereas a single 15% validation split is a noisier basis for publishing weights.

Override either way:

```bash
REGISTER_BEST_WEIGHTS=1 ./scripts/hyperparameter/tune_hpo_holdout3.sh --model oped ...
REGISTER_BEST_WEIGHTS=0 ./scripts/hyperparameter/tune_hpo_cv5.sh --model oped ...
```

## Usage

```bash
./scripts/hyperparameter/check_tuning_status.sh pridict2
./scripts/hyperparameter/check_tuning_status.sh pridict2 minsepie/library_insert_set12/hek293t/pe2

./scripts/hyperparameter/tune_hpo_cv5.sh --model pridict2 \
  --dataset-name demo --study minsepie --dataset library-insert-set12 \
  --cell-line hek293t --pe-system pe2

# Idempotent re-run: exits early if a preset already exists
SKIP_IF_TUNED=1 ./scripts/hyperparameter/tune_hpo_cv5.sh --model pridict2 \
  --dataset-name demo --study minsepie --dataset library-insert-set12 \
  --cell-line hek293t --pe-system pe2
```

## Artifacts

| What | Where |
|---|---|
| HPO result presets (gitignored) | `services/pe-ensemble/config/training_presets_local/<model>.yaml` |
| Shipped defaults | [`services/pe-ensemble/config/training_presets/`](../../services/pe-ensemble/config/training_presets/README.md) |
| Optuna storage (resumable) | `services/pe-ensemble/tuning_studies/*.db` |
| Tuning job state and logs | `services/pe-ensemble/tune_jobs/<job_id>/` — see [`jobs/README.md`](../../services/pe-ensemble/jobs/README.md) |

Search spaces are defined per model in
`services/pe-ensemble/app/training/search_spaces.py`; how a preset resolves
against request hyperparameters is described in the training presets README.

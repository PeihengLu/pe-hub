# Experiment scripts (dataset-specific HPO recipes)

Shared helpers (`check_tuning_status`, `tune_hpo_cv5`, `_common`) live in
[`../hyperparameter/`](../hyperparameter/README.md).

Data loading goes through **`peen` → `pe_db.library`** (same filter/merge/split path as
[`pedb filter`](../../services/pe-db/README.md#filter-and-export); no PE-DB HTTP server).

## How these scripts are organized

Three layers, so a run can be traced from the shell command down to the code:

1. **Recipes** — the scripts in this directory. Each pins one model × dataset ×
   protocol and nothing else.
2. **Generic runners** — [`../hyperparameter/`](../hyperparameter/README.md)
   (`tune_hpo_cv5.sh`, `tune_hpo_holdout3.sh`) plus `_common.sh` for defaults and
   `SKIP_IF_TUNED` handling.
3. **CLI and service code** — `peen train` / `peen tune` / `peen evaluate`, which
   are the same runners the HTTP API uses
   (`services/pe-ensemble/pe_ensemble/training/runner.py`,
   `pe_ensemble/training/tune_runner.py`, `pe_ensemble/evaluation/runner.py`).

Multi-step experiments get their own subdirectory with numbered stage scripts and
a `run_all.sh`: [`pridict2-reproduction/`](pridict2-reproduction/README.md),
[`scratch-benchmark/`](scratch-benchmark/README.md),
[`datasheet-benchmark/`](datasheet-benchmark/README.md).

Runs are seeded (`SPLIT_RANDOM_STATE` for splits, the training `seed`
hyperparameter for the model), so re-running a recipe with the same inputs
reproduces it. The scratch and datasheet benchmarks vary the seed deliberately
across repeats to measure run-to-run spread.

## Default split protocol

1. **Most datasets** — random group split:

   - outer test holdout (`TEST_PCT`, default `0.15`)
   - 5-fold CV on the remainder (`CV_FOLDS=5`)
   - Optuna maximizes the mean fold validation metric
   - PRIDICT author `testset_fold` is **not** used (library1 has none)
2. **DeepPrime only** — author folds via `--use-original-fold`:

   - `original_fold = -1` → permanent test
   - `original_fold ∈ {0..4}` → CV folds
3. **Merged DeepPrime ClinVar + PRIDICT library1** — `--merge --use-original-fold`:
   PE-DB concatenates sheets, then `propagate_original_fold_by_target_uid` copies
   DeepPrime folds onto library1 rows that share a protospacer/`target_uid`.
   Non-overlapping library1 loci get random CV (+ optional outer test).
   See [pe-db filter docs](../../services/pe-db/README.md#splits-author-folds-and-merge).

Prerequisites: `./scripts/setup-python-env.sh` then `conda activate pe-hub && ./scripts/install-clis.sh`

Smoke: `SMOKE=1 DEVICE=mps ./scripts/experiments/<script>.sh`

## PRIDICT 2.0 reproduction (transfer + ensemble)

Full pipeline (tune → base train → fine-tune → mean ensemble):

```bash
./scripts/experiments/pridict2-reproduction/run_all.sh
```

See [`pridict2-reproduction/README.md`](pridict2-reproduction/README.md).

## Base model evaluation (pooled benchmarks)

Cross-benchmark evaluation of base vendor weights with leak prevention on:

- **Weights:** `DeepPrime_base`, OPED merged, OptiPrime `base`, and the August 2023
  PRIDICT2 **mean ensemble** (Model A `pridict1_1` + Model B `pridict1_2` at the
  matching CV fold and HEK/K562 head). Single A/B checkpoints are not scored.
  `hek` and `hek293t` are the same line. Override with `PRIDICT2_HEADS=HEK`
  (or `K562`) to score one head.
- **Benchmarks:** MinSePIE insert (pooled libraries), DeepPE (pooled assays), DeepPrime ClinVar,
  PRIDICT library1, PRIDICT library-diverse, OptiPrime lib-mmr, OptiPrime lib-cv.
  Each **cell line** is scored separately (library-diverse HEK293T / K562 / K562MLH1dn,
  DeepPE HEK293T vs HCT116 vs MDA-MB-231, OptiPrime HEK293T vs HeLa, …).
  OptiPrime Lib-MMR / Lib-CV also split **PE2 vs PE4** (`…__hek293t__pe2`).
- **Splits:** DeepPrime ClinVar and DeepPE use the author `original_fold=-1`
  test set (Kim et al. HT / type / position tests; endo has no author fold and
  is not mixed into that test). PRIDICT2 **`run_x` tests library-diverse
  `testset_fold==x`** (the fold that checkpoint held out). Other benches use a
  random group holdout. **PRIDICT library1 has no author test split**
  (`original_fold` is unset). Vendor models trained on that sheet (PRIDICT2 A/B,
  OptiPrime) record **all** library1 loci as training data, so in-domain
  library1 eval is `data_leak`. DeepPrime / OPED were not trained on library1
  and can still be scored there.

```bash
conda activate pe-hub
# Optional: backfill vendor train_target_loci
# DeepPrime/OPED: author train folds only. PRIDICT2: all library1 (+ ClinVar
# train folds for Model B) and library-diverse minus the held-out fold.
# OptiPrime: all library1 + library-diverse + lib-* + ClinVar (Hsu pooled
# protospacer CV; has_original_test_split is false, so in-domain eval aborts).
cd services/pe-ensemble
python -m pe_ensemble.models.deepprime_vendor_provenance
python -m pe_ensemble.models.oped_vendor_provenance
python -m pe_ensemble.models.optiprime_vendor_provenance
python -m pe_ensemble.models.pridict2_vendor_provenance
cd ../..

DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Script invokes ``python -m pe_ensemble.cli`` (more reliable than the peen entrypoint).
python scripts/experiments/summarize_eval_results.py results/base_model_eval/<RUN_ID>/results.jsonl
python scripts/experiments/plot_base_model_eval.py \
  results/base_model_eval/<RUN_ID>/paper_comparison.csv
# Writes txt/diagrams/eval_pearson_heatmap.pdf (and .png) using the same
# Tableau palette as data_composition.png. Pass --all-figures for bars too.

# Partial rerun: reuse RUN_ID so new cells replace matching rows, then summary.csv
# is rewritten. Skip DeepPrime (already good); OptiPrime lib-* data_leak rows stay.
DEVICE=cuda:0 MODELS=oped,pridict2,optiprime RUN_ID=<RUN_ID> \
  ./scripts/experiments/evaluate_base_model_benchmarks.sh
# PRIDICT2 ensembles only (both heads). HEK-only:
DEVICE=cuda:0 MODELS=pridict2 PRIDICT2_HEADS=HEK RUN_ID=<RUN_ID> \
  ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Re-score library-diverse ensembles with fold-matched splits:
DEVICE=cuda:0 MODELS=pridict2 BENCHMARKS=pridict2-library-diverse RUN_ID=<RUN_ID> \
  SKIP_EXISTING=1 ./scripts/experiments/evaluate_base_model_benchmarks.sh
# OptiPrime-only, non-leak benches. If a prior run marked OptiPrime
# ``cli_failure`` but logs show success (vendor ``syn{50}`` in stdout), repair:
python scripts/experiments/summarize_eval_results.py \
  results/base_model_eval/<RUN_ID>/results.jsonl --repair-from-logs
```

Outputs under `results/base_model_eval/<RUN_ID>/`:

- `results.jsonl` — one record per evaluation (including `data_leak` aborts)
- `summary.csv` — flat table for plotting
- `summary_cv_mean_std.csv` — PRIDICT2 experiment × head × benchmark mean±std across folds

Latest completed run id is also written to `results/base_model_eval/LATEST_RUN_ID`.
The `results/` tree is gitignored and tracked with DVC (`results.dvc`); push/pull
via the ARC remote (see the Oxford ARC README).

**Notes from the reference run:**
- Vendor provenance for DeepPrime / OPED records **train folds only** (author
  `Test` / `original_fold=-1` excluded). PRIDICT2 records **all library1
  loci** (no author split) plus library-diverse minus `run_x`, and Model B
  also includes ClinVar train folds. OptiPrime records pooled Hsu training
  sheets with `has_original_test_split: false`. Sync with
  `python -m pe_ensemble.models.deepprime_vendor_provenance`,
  `python -m pe_ensemble.models.oped_vendor_provenance`,
  `python -m pe_ensemble.models.optiprime_vendor_provenance`, and
  `python -m pe_ensemble.models.pridict2_vendor_provenance` from `services/pe-ensemble`.
- Partial train/test locus overlap on an **author holdout this weight used**
  excludes overlapping `target_uid`s and continues. Weights with
  `has_original_test_split: false` (OptiPrime `base`) abort in-domain eval
  with `no_original_test_split` even when the sheet has another paper's
  `original_fold` (library-diverse, ClinVar). Full overlap (e.g. OptiPrime ×
  lib-mmr/lib-cv, or PRIDICT2/OptiPrime × library1) still aborts as `data_leak`
  unless `--allow-data-leak`. Hsu's in-domain Pearson 0.723 is filled only on
  lib-mmr / lib-cv leak cells; library-diverse and ClinVar stay `leak_unfilled`.
- OptiPrime needs the JAX stack (`jax`, `flax`, `chex`, …). Installed automatically by `./scripts/install-clis.sh` on Python 3.11.

Smoke: `SMOKE=1 DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh`

## Scratch benchmark (cross-model, 10 trials × 3 seeds)

One GPU job per seed: Optuna + `register_best_weights` + eval (DeepPrime, OPED,
PRIDICT2) via [`datasheet-benchmark`](datasheet-benchmark/README.md) on the same
seven pooled datasets as the base-model eval.

```bash
conda activate pe-hub
SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh
./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh   # 63 short L40S jobs
```

See [`scratch-benchmark/README.md`](scratch-benchmark/README.md).

## Datasheet benchmark (nested Optuna)

Generic from-scratch benchmark for **one dataset or datasheet** with optional
`--edit-type` filters. **N** (folds or random seeds) and **X** (Optuna trials
per fold/seed) are required.

- **Small** sheets (`n_rows < 50k` by default): N-fold CV, independent Optuna
  search on each outer fold
- **Large** sheets: holdout_3 (70/15/15) repeated N times with distinct split
  and initialization seeds

```bash
./scripts/experiments/datasheet-benchmark/run.sh \
  --model pridict2 --n 5 -x 20 \
  --study minsepie --dataset library-insert-set12 \
  --cell-line hek293t --pe-system pe2 \
  --edit-type ins
```

See [`datasheet-benchmark/README.md`](datasheet-benchmark/README.md).

## From-scratch train probe

Lightweight sequential from-scratch trains (`load_pretrained=false`) for DeepPrime,
OPED, and PRIDICT2 on library1, library-diverse, and DeepPrime ClinVar. Streams
peen output and prints each job's full `train.log`.

```bash
conda activate pe-hub
DEVICE=mps ./scripts/experiments/probe_scratch_train.sh
SMOKE=1 DEVICE=mps ./scripts/experiments/probe_scratch_train.sh
MODELS=oped DATASET_NAMES=pridict1-library1 DEVICE=mps \
  ./scripts/experiments/probe_scratch_train.sh

# Targeted smoke probes (NUM_WORKERS defaults to 15)
SMOKE=1 DEVICE=cuda:0 MODELS=pridict2 DATASET_NAMES=pridict1-library1 \
  NUM_WORKERS=15 ./scripts/experiments/probe_scratch_train.sh
SMOKE=1 DEVICE=cuda:0 MODELS=oped DATASET_NAMES=deepprime-clinvar \
  NUM_WORKERS=15 ./scripts/experiments/probe_scratch_train.sh
```

## Entry points at a glance

| Path | Purpose |
| --- | --- |
| `evaluate_base_model_benchmarks.sh` | Pooled evaluation of vendor base weights with leak prevention |
| `probe_scratch_train.sh` | Quick sequential from-scratch trains |
| [`pridict2-reproduction/run_all.sh`](pridict2-reproduction/README.md) | PRIDICT 2.0 transfer + ensemble reproduction |
| [`scratch-benchmark/run_all.sh`](scratch-benchmark/README.md) | Cross-model from-scratch matrix (tune → train → evaluate) |
| [`datasheet-benchmark/run.sh`](datasheet-benchmark/README.md) | Nested Optuna benchmark for one dataset or datasheet |

One-off per-model tuning is done by calling the generic runners directly with
the dataset flags, rather than by a dedicated script per dataset:

```bash
./scripts/hyperparameter/check_tuning_status.sh pridict2 minsepie/library_insert_set12/hek293t/pe2
SKIP_IF_TUNED=1 ./scripts/hyperparameter/tune_hpo_cv5.sh --model pridict2 \
  --dataset-name minsepie-insert --study minsepie --dataset library-insert-set12 \
  --cell-line hek293t --pe-system pe2
```

## Analysis helpers

Python utilities used by the shell scripts, and runnable on their own against a
finished run:

| Script | Purpose |
| --- | --- |
| `summarize_eval_results.py` | `results.jsonl` → `summary.csv`; `--repair-from-logs` recovers mislabelled `cli_failure` rows |
| `plot_base_model_eval.py` | Pearson heatmap (and `--all-figures` bars) into `txt/diagrams/` |
| `expand_eval_cell_lines.py` | Expands a benchmark into its per-cell-line, per-PE-system cells |
| `eval_split_args.py` | Chooses author-fold versus random-holdout split flags per benchmark |
| `paper_reported_metrics.py` | Published reference metrics for the comparison table |

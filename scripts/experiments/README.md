# Experiment scripts (dataset-specific HPO recipes)

Shared helpers (`check_tuning_status`, `tune_hpo_cv5`, `_common`) live in
[`../hyperparameter/`](../hyperparameter/README.md).

Data loading goes through **`peen` → `pe_db.library`** (same filter/merge/split path as
[`pedb filter`](../../services/pe-db/README.md#filter-and-export); no PE-DB HTTP server).

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

- **Weights:** `DeepPrime_base`, OPED merged, OptiPrime `base`, and August 2023
  PRIDICT2 CV folds (`pridict1_{1,2}__exp_2023-08-*__run_{0..4}`). **Model A**
  (`pridict1_1`) is library1-base fine-tuned on library-diverse; **Model B**
  (`pridict1_2`) is library1+ClinVar-base fine-tuned on library-diverse. Each
  run has both `__HEK` and `__K562` heads. `hek` and `hek293t` are the same
  line. Override with `PRIDICT2_HEADS=HEK` (or `K562`) to score one head.
- **Benchmarks:** MinSePIE insert (pooled libraries), DeepPE (pooled assays), DeepPrime ClinVar,
  PRIDICT library1, PRIDICT library-diverse, OptiPrime lib-mmr, OptiPrime lib-cv.
  Each **cell line** is scored separately (library-diverse HEK293T / K562 / K562MLH1dn,
  DeepPE HEK293T vs HCT116 vs MDA-MB-231, OptiPrime HEK293T vs HeLa, …).
- **Splits:** DeepPrime ClinVar uses the author `original_fold=-1` test set.
  PRIDICT2 **`run_x` tests library-diverse `testset_fold==x`** (the fold that
  checkpoint held out). Other benches use a random group holdout.
  **PRIDICT library1 has no author test split** (`original_fold` is unset).
  Vendor models trained on that sheet (PRIDICT2 A/B, OptiPrime) record **all**
  library1 loci as training data, so in-domain library1 eval is `data_leak`.
  DeepPrime / OPED were not trained on library1 and can still be scored there.

```bash
conda activate pedb
# Optional: backfill vendor train_target_loci
# DeepPrime/OPED: author train folds only. PRIDICT2: all library1 (+ ClinVar
# train folds for Model B) and library-diverse minus the held-out fold.
# OptiPrime: all library1 + ClinVar train folds + library-diverse + lib-*.
cd services/pe-ensemble
python -m app.models.deepprime_vendor_provenance
python -m app.models.oped_vendor_provenance
python -m app.models.optiprime_vendor_provenance
python -m app.models.pridict2_vendor_provenance
cd ../..

DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Script invokes ``python -m pe_ensemble.cli`` (more reliable than the peen entrypoint).
python scripts/experiments/summarize_eval_results.py results/base_model_eval/<RUN_ID>/results.jsonl

# Partial rerun: reuse RUN_ID so new cells replace matching rows, then summary.csv
# is rewritten. Skip DeepPrime (already good); OptiPrime lib-* data_leak rows stay.
DEVICE=cuda:0 MODELS=oped,pridict2,optiprime RUN_ID=<RUN_ID> \
  ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Both PRIDICT2 heads on every sheet (default). HEK-only:
DEVICE=cuda:0 MODELS=pridict2 PRIDICT2_HEADS=HEK RUN_ID=<RUN_ID> \
  ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Re-score library-diverse with fold-matched splits (SKIP_EXISTING will not
# skip older random-holdout rows of the same weight):
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
  also includes ClinVar train folds. Sync with
  `python -m app.models.deepprime_vendor_provenance`,
  `python -m app.models.oped_vendor_provenance`,
  `python -m app.models.optiprime_vendor_provenance`, and
  `python -m app.models.pridict2_vendor_provenance` from `services/pe-ensemble`.
- Partial train/test locus overlap excludes overlapping `target_uid`s from the
  test partition and continues; full overlap (e.g. OptiPrime × lib-mmr/lib-cv,
  or PRIDICT2/OptiPrime × library1) still aborts as `data_leak` unless
  `--allow-data-leak`.
- OptiPrime needs the JAX stack (`jax`, `flax`, `chex`, …). Installed automatically by `./scripts/install-clis.sh` on Python 3.11.

Smoke: `SMOKE=1 DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh`

## Scratch benchmark (cross-model, holdout_3 HPO)

Tune → train → evaluate matrix for DeepPrime, OPED, and PRIDICT2 on library1,
library-diverse, and DeepPrime ClinVar (same benchmarks as the probe script).
ARC-ready via `submit.sh`.

```bash
conda activate pe-hub
SMOKE=1 DEVICE=cuda:0 ./scripts/experiments/scratch-benchmark/run_all.sh
./scripts/cluster/oxford-arc/submit.sh 01_tune_matrix.sh   # on ARC
```

See [`scratch-benchmark/README.md`](scratch-benchmark/README.md).

## From-scratch train probe

Lightweight sequential from-scratch trains (`load_pretrained=false`) for DeepPrime,
OPED, and PRIDICT2 on library1, library-diverse, and DeepPrime ClinVar. Streams
peen output and prints each job's full `train.log`.

```bash
conda activate pedb
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

## Other recipes

| Script                                 | Purpose                                       |
| -------------------------------------- | --------------------------------------------- |
| `tune_pridict2_minsepie.sh`          | PRIDICT2 on MinSePIE`library-insert-set12`  |
| `tune_oped_deeppe_ht.sh`             | OPED on DeepPE HT                             |
| `tune_deepprime_author_folds.sh`     | DeepPrime ClinVar with**author** folds  |
| `tune_pridict2_merged_l1_clinvar.sh` | → redirect to`pridict2-reproduction/02_…` |
| `tune_pridict2_library_diverse.sh`   | → redirect to`pridict2-reproduction/05_…` |

```bash
./scripts/hyperparameter/check_tuning_status.sh pridict2
SKIP_IF_TUNED=1 ./scripts/experiments/tune_pridict2_minsepie.sh
```

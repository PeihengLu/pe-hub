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
   - PRIDICT author `testset_fold` is **not** used
2. **DeepPrime only** — author folds via `--use-original-fold`:

   - `original_fold = -1` → permanent test
   - `original_fold ∈ {0..4}` → CV folds
3. **Merged DeepPrime ClinVar + PRIDICT library1** — `--merge --use-original-fold`:
   PE-DB concatenates sheets, then `propagate_original_fold_by_target_uid` copies
   DeepPrime folds onto library1 rows that share a protospacer/`target_uid`.
   Non-overlapping library1 loci get random CV (+ optional outer test).
   See [pe-db filter docs](../../services/pe-db/README.md#splits-author-folds-and-merge).

Prerequisites: `conda activate <env> && ./scripts/install-clis.sh`

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
  PRIDICT2 CV folds with the HEK head
  (`pridict1_{1,2}__exp_2023-08-*__run_{0..4}__HEK`). December 2023 PRIDICT2
  experiments are excluded (incomplete bundles; see notes).
- **Benchmarks:** MinSePIE insert (pooled), DeepPE (pooled), DeepPrime ClinVar,
  PRIDICT library1, PRIDICT library-diverse, OptiPrime lib-mmr, OptiPrime lib-cv

```bash
conda activate pedb
# Optional: backfill vendor train_target_loci (train folds only for DeepPrime/OPED)
cd services/pe-ensemble
python -m app.models.deepprime_vendor_provenance
python -m app.models.oped_vendor_provenance
python -m app.models.optiprime_vendor_provenance
cd ../..

DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh
# Script invokes ``python -m pe_ensemble.cli`` (more reliable than the peen entrypoint).
python scripts/experiments/summarize_eval_results.py results/base_model_eval/<RUN_ID>/results.jsonl
```

Outputs under `results/base_model_eval/<RUN_ID>/`:

- `results.jsonl` — one record per evaluation (including `data_leak` aborts)
- `summary.csv` — flat table for plotting
- `summary_cv_mean_std.csv` — PRIDICT2 experiment × benchmark mean±std across folds

Latest completed run id is also written to `results/base_model_eval/LATEST_RUN_ID`.
The `results/` tree is gitignored and tracked with DVC (`results.dvc`); push/pull
via the ARC remote (see the Oxford ARC README).

**Notes from the reference run:**
- PRIDICT2 December 2023 experiments are dropped from the eval matrix: config
  expects `K562MLH1dn` heads that were never packaged, and vendor sources cannot
  be remigrated. They remain in the weights registry as known-broken.
- Vendor provenance for DeepPrime / OPED records **train folds only** (author
  `Test` / `original_fold=-1` excluded). Sync with
  `python -m app.models.deepprime_vendor_provenance` and
  `python -m app.models.oped_vendor_provenance` from `services/pe-ensemble`.
- Partial train/test locus overlap excludes overlapping `target_uid`s from the
  test partition and continues; full overlap (e.g. OptiPrime ensemble ×
  lib-mmr/lib-cv) still aborts as `data_leak` unless `--allow-data-leak`.
- OptiPrime needs JAX stack deps (`jax`, `flax`, `chex`, …).

Smoke: `SMOKE=1 DEVICE=mps ./scripts/experiments/evaluate_base_model_benchmarks.sh`

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

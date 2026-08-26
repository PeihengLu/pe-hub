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

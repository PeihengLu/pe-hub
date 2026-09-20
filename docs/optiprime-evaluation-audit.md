# OptiPrime evaluation audit — 20 September 2026

The wrapper's five-checkpoint inference agrees with the author's entrypoint,
but the PE-DB input conversion and evaluation metadata contained errors. This
audit fixes the confirmed errors; it does not reproduce the paper's benchmark.

Reference: [Hsu's original repository](https://github.com/alvin-hsu/optiprime-src),
particularly the vendored `DESIGN_PE.py`, `PREDICT_PE.py`,
`scripts/pe/pe_inputs.py`, and `scripts/pe/pe_datasets.py`.
The before-converter snapshot is from PE-Hub commit
`b7472611a19f4ad9938468449db43ce340dae96c`.

## Corrections

- Target windows now end four actual bases after the aligned RTT endpoint,
  matching `DESIGN_PE`'s `edited_seq[:25 + rtt_len]` and corresponding WT
  slice. Previously, the entire standardized window changed the learned
  homology-length and terminal-dinucleotide features.
- The genomic 20-mer is anchored to the PAM-proximal spacer endpoint after
  accounting for alignment pads. This handles reported 19/21-base spacers and
  insertions within the spacer. A separate genomic `proto30` preserves Rule
  Set 3 context when the modeled WT window is shorter than 30 bases.
- OptiPrime-format labels from DeepPE, DeepPrime, PRIDICT1/2, Anzalone and
  MinSePIE are converted from percentage points to fractions. Hsu's labels and
  ad-hoc inputs retain their fraction convention. This uses study metadata,
  not an unreliable maximum-label heuristic. Predictions remain fractions.
- Prediction preprocessing uses dummy observations and positive weights so
  the vendor's training-data filters cannot discard unlabeled/zero-weight
  input rows. Assay fields and the author's one-day time adjustment remain.
- Unknown weight IDs now raise instead of silently loading `base`.
- ClinVar's DeepPrime test folds are included in OptiPrime's pooled training
  provenance. The existing prohibition on treating the five-fold ensemble's
  training datasets as independent tests remains. After refreshing the Hsu
  standardized data, the rebuilt sidecar contains 88,047 target UIDs.
- OptiPrime formatted-cache revision is now 7, invalidating old conversions.

## Validation

**Author parity:** six Lib-MMR examples, all five checkpoints, batch size four
(including a partial last batch). Maximum absolute difference between the
wrapper and the original `PREDICT_PE.main` was **4.6303e-9**. Both received the
same converted inputs and the existing scalar-output compatibility patch.
This establishes inference parity, not independent validation of missing
sequence context or a held-out performance result.

**External diagnostic:** all 31 measured DeepPE endogenous HEK293T examples
in the refreshed sheet, all five checkpoints, batch size eight. Their target
UIDs had zero overlap with the rebuilt training sidecar. Both before and after
used the same refreshed standardized rows and fraction-scale observations;
the before run used the saved original converter with the current wrapper.

| Metric | Before conversion fix | After conversion fix |
| --- | ---: | ---: |
| Pearson | 0.419926 | 0.492624 |
| Spearman | 0.653695 | 0.618611 |
| MSE (fraction squared) | 0.004977 | 0.010480 |
| MAE (fraction) | 0.051448 | 0.077544 |

Pearson improved, while rank correlation and absolute calibration worsened.
The change restores the author's input semantics; it is not a demonstrated
general accuracy improvement. The old percentage/fraction mismatch yielded
MSE 88.50285 and MAE 5.51364 on this same set, which were not meaningful
fraction-scale errors.

**Tests:** 41 conversion/cache tests and 32 wrapper/provenance/leakage tests
passed. A broader conversion run also failed three existing author-data
comparisons for PRIDICT2, DeepPrime ClinVar and OPED/DeepPE-HT. Their converter
implementations and source sheets were not changed in this audit. That broader
suite is not fully green.

## Data and runtime qualifications

- The local Hsu parquet files were stale: for example, Lib-MMR still carried
  the old four-base spacer offset. All 12 Hsu sheets and the DeepPE endogenous
  HEK293T sheet were regenerated using the existing standardizers. Two
  unmeasured DeepPE rows were removed by that standardizer.
- Lib-MMR target strings can lack upstream genomic bases. The existing `A`
  padding remains an approximation. Across its four assay sheets, 649, 645,
  682 and 682 rows respectively also lack four downstream bases. The converter
  now warns about incomplete context rather than implying exact reconstruction.
- Existing substitutions of a trained lab/cell group for an unseen group
  remain extrapolations; this audit did not establish their calibration.
- Native inference needed temporary dependencies: SciPy 1.14.1, NumPy 2.2.6,
  TensorFlow 2.20.0 and protobuf 5.29.6, with pytest installed temporarily.
  The installed SciPy 1.15 binary failed to load on this Mac, and installed
  TensorFlow 2.16.2 conflicted with the NumPy/JAX stack. `OMP_NUM_THREADS=1`
  avoided a native crash during Rule Set 3 features. Tiny audit runs used
  the vendor's serial feature pool. The conda environment was not modified.

Results, prediction CSVs, logs, the original converter, and `reproduce.py` are
in `results/optiprime_audit/20260920/`. From the repository root, the tested
temporary environment can reproduce the numerical checks with:

```sh
MPLCONFIGDIR=/tmp/pehub-mpl XDG_CACHE_HOME=/tmp/pehub-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/tmp/pehub-optiprime-scipy114:/tmp/pehub-optiprime-tf \
conda run --no-capture-output -n pe-hub \
python results/optiprime_audit/20260920/reproduce.py
```

The temporary dependency directories must still exist. Existing benchmark
results were not overwritten; cached historical scores need reevaluation to
reflect these changes.

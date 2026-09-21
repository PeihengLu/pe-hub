# OPED training audit — 21 September 2026

## Impact on previous results

PE-Hub's scratch constructor instantiated `TransformerEncoderModelOrder3`.
The vendored `train_and_test_transformer_order3` instead instantiates
`TransformerEncoderDecoderModelOrder3`, with PBS/RT decoder branches attending
to the target encodings. This was an integration error, not simply different
hyperparameters. The scratch default is now encoder–decoder. The old class
remains available only through explicit `model_variant="encoder"` for labeled
legacy experiments. Loading an existing encoder checkpoint still recognizes
it as an encoder; it is not silently converted into a decoder.

Prior scratch runs measure an encoder-only variant, not the intended OPED
architecture. Paper-aligned scratch comparisons require rerunning those OPED
trials and final training. Their weights cannot be relabeled as trained
encoder–decoder weights. Fine-tuning the shipped checkpoint used its decoder
architecture and was not affected by this constructor error. Pretrained
evaluation and sequence-context conversion were not changed.

Only the shipped OPED weight entry was found locally, apart from this audit's
new artifacts. This does not identify the architecture of unavailable remote
checkpoints. Tensor names beginning `encoder.` versus `encoder_decoder.` can
distinguish old saved models without relying on their filenames.

The OPED HPO space now explicitly fixes `model_variant="encoder_decoder"`.
This changes the study fingerprint so the normal tuning path does not resume
encoder-only trials in a decoder study. Old optimized presets/results remain
historical artifacts, not decoder-specific tuning results.

## Other confirmed errors fixed

- **Best-checkpoint selection:** shared Lightning history averaged batch losses
  equally. It now weights by sample count, matching early stopping. A regression
  test demonstrates the old code choosing the wrong epoch when the final batch
  is smaller. Equal-sized validation batches were unaffected by this issue.
- **Epoch logs:** training loss was read before Lightning finalized it, yielding
  a missing first-epoch value and subsequent values delayed by one epoch.
  History now aggregates the current epoch's training batches.
- **Hyperparameters:** scratch defaults and preset layers could shadow explicit
  `epochs`, `dropout`, `ffn_dim`, `encoder_layers`, and `ntoken` aliases.
  Aliases now resolve within each layer before merging. The explicit
  `freezing=True` train argument is also honored.
- **Checkpoint fidelity:** tensor shapes do not encode attention-head count or
  dropout. New saves include a `*.architecture.json` sidecar and loaders use it.
  This prevents custom heads from silently changing during reload. Keep the
  sidecar with its weights file. Legacy vendor checkpoints retain their existing
  loading behavior; missing metadata in old custom runs cannot be recovered
  from tensors alone. Saving to a basename and loading registry directories work.
- **Fine-tuning reports:** effective architecture now reflects the loaded model,
  rather than reporting scratch defaults such as eight heads and six layers for
  a 64-head, one-layer pretrained decoder.

## Validation

The corrected optimizer step matches an explicit Adam/MSE/gradient-clipping
step exactly. Tests also cover frozen-backbone updates, partial validation
batches, short all-padding k-mer channels, alias precedence through API presets,
decoder construction, and prediction-preserving checkpoint reloads.

**79 tests passed, one skipped:** 66 focused OPED/shared-helper/preset/loading
checks plus 13 DeepPrime reproducibility checks for the modified shared loop.
The latter initially hit sandbox restrictions on PyTorch worker processes and
passed when run outside that sandbox. The tests used the `pe-hub` interpreter
with the previous audit's temporary SciPy/NumPy/pytest dependencies; the installed
conda environment was not modified.

A small real-data scratch smoke run used 96 training and 32 validation DeepPE
HT examples, one per `group_id`, with disjoint group IDs. It used the existing
sequence conversion, embedding size 16, four heads, FFN size 32, one layer,
zero dropout, Adam LR 0.003, batch size 16, and 20 epochs. This is a training
sanity check, not the published model configuration or a benchmark reproduction.

| Corrected encoder–decoder run | Initial | Restored best checkpoint |
| --- | ---: | ---: |
| Training MSE | 230.1111 | 70.7368 |
| Validation MSE | 154.3655 | 47.0456 |

Reloaded predictions matched exactly (maximum absolute difference zero).
A separate two-epoch run fine-tuned the actual shipped 64-head decoder on eight
training/eight validation examples: the frozen embedding was unchanged and the
output head changed. A regression test additionally checks every frozen backbone
parameter in a small decoder.

Results and reproduction script: `results/oped_training_audit/20260921/`.
`encoder_only_metrics.json` preserves the initial diagnostic made before the
architecture correction; `metrics.json` records the corrected decoder run.

Two protocol differences remain explicit: PE-Hub selects epochs by validation
MSE, whereas the vendored routine selects by Pearson; its default StepLR decays
by 0.95 every ten epochs, whereas the vendor routine uses gamma 1. Architecture
correction alone does not reproduce every published training setting. No full
benchmark rerun or historical-result replacement was performed.

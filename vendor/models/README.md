# State of the art models

## Unified PyTorch environment

These vendor models historically targeted different PyTorch versions
(DeepPrime ≈ 1.10–1.11, OPED ≈ 1.6–1.8, PRIDICT2 = 2.0.1). They are now run
together in a **single environment** pinned to `torch>=2.0,<2.9` (see
`services/pe-ensemble/pyproject.toml`).

This works because every model loads its weights as a **`state_dict`** (a plain
dict of tensors keyed by layer name), which is decoupled from the PyTorch
version and code layout:

- **DeepPrime** — `state_dict` files (`models/.../*.pt`).
- **PRIDICT2** — `state_dict` files (`trained_models/.../model_statedict/*.pkl`).
- **OPED** — `state_dict` file
  `oped/pegRNA_PredictingCodes/Model_Trained/pegRNA_Model_Merged_saved.order3_decoder_weights.pt`.

`services/pe-ensemble/tests/test_weights_loading.py` loads all three under the
pinned torch as a regression guard.

### OPED legacy full-pickle weights (do not load directly)

OPED also ships two **full-pickle** checkpoints
(`...order3_decoder.pt` and `...order3_decoder_torch2.pt`). These pickle the
entire `nn.Module` and embed the original module paths, so they fail to load
on PyTorch ≥ 2.0 and across refactors. They are kept only because OPED's own
Django app (`oped/pegRNA/utils.py`) still references them in its original
environment. **The PE-ensemble OPED wrapper refuses to load them** and defaults
to the `*_weights.pt` state_dict. To regenerate the state_dict from a
full-pickle, run:

```bash
cd services/pe-ensemble
python -m app.models.convert_oped_weights \
  ../../vendor/models/oped/pegRNA_PredictingCodes/Model_Trained/pegRNA_Model_Merged_saved.order3_decoder.pt \
  ../../vendor/models/oped/pegRNA_PredictingCodes/Model_Trained/pegRNA_Model_Merged_saved.order3_decoder_weights.pt
```

## Inferencing using the pre-trained models

### DeepPrime



### PRIDICT and PRIDICT 2.0



### OPED
- For inferencing, OPED only requires three columns in a dataframe:
  - 'Target(47bp)'
  - 'PBS'
  - 'RT'
- Path to the trained weights:
  - `/home/peiheng/development/pe-db/models/OPED/pegRNA_PredictingCodes`
- The pickled model was training using pytorch 1.18.1, compatibility issue will occur with pytorch 2.0.0 and above.


# PE Common

Shared utilities package for PE Database and PE Ensemble services.

Anything both services must agree on lives here — device naming, split
assignment, the standardized-sequence conventions, the training loop, and the
plugin contract. Everything else stays in the service that owns it.

## Installation

```bash
pip install -e packages/pe-common
```

## Module map

| Module | Purpose | Import cost |
|---|---|---|
| `constants.py` | Repository paths and the default device | free |
| `devices.py` | Device discovery and `"auto"` / `"cuda:0"` resolution | free |
| `splits.py` | Train/val/test assignment; the one source of split semantics | free |
| `sequence_utils.py` | WT/mutant alignment and padding | free |
| `data_utils.py` | Evaluation partition masks | free |
| `cell_lines.py` | Cell-line name normalization across studies | free |
| `model_interface.py` | The `BasePEModel` contract | free |
| `plugins.py` | Plugin manifest discovery and loading | free |
| `plugin_validation.py` | Validates a plugin against the contract | free |
| `conversion_progress.py` | Progress callbacks for long conversions | free |
| `filter_params.py` | Catalog/edit filter field names for CLI, HTTP, and `filter_from_params` | free |
| `training.py` | Training loops, Lightning glue, seeding, metrics | needs PyTorch (lazy) |
| `features.py` | MFE, melting temperature, GC content | needs ViennaRNA (lazy) |

The two lazy modules are re-exported through the `pe_common` namespace so
importing `pe_common` never pulls in PyTorch or ViennaRNA on its own.

## Contents

### Constants (`pe_common.constants`)

| Name | Description |
|------|-------------|
| `PROJECT_ROOT` | Repository root (env override: `PE_PROJECT_ROOT`) |
| `DATA_ROOT` | `datasets/` directory |
| `MODEL_ROOT` | `vendor/models/` directory |
| `DEVICE` | Default PyTorch device (`mps` / `cuda` / `cpu`) |

### Devices (`pe_common.devices`)

Discover and resolve compute devices for training and inference:

```python
from pe_common.devices import list_devices, resolve_device, default_device_id

for device in list_devices():
    print(device.device_id, device.name)

torch_device = resolve_device("cuda:0")  # or "auto", "mps", "cpu"
```

### Splits (`pe_common.splits`)

Train/validation/test assignment shared by PE-DB export and Ensemble training:

```python
from pe_common.splits import SplitConfig, assign_splits, split_config_from_params

config = split_config_from_params(
    strategy="holdout_3",
    train_pct=0.7,
    val_pct=0.15,
    test_pct=0.15,
)
df = assign_splits(df, config, group_col="group_id")
```

Strategies: `none`, `holdout_2`, `holdout_3`, `cv`. Supports author `original_fold`
columns and grouped k-fold by `group_id`.

### Filter params (`pe_common.filter_params`)

The catalog/edit query keys shared by `pedb filter`, `peen` train/eval flags,
and PE-DB `filter_from_params` (HTTP `GET /api/filter` uses the same names):

```python
from pe_common.filter_params import FILTER_LIST_FIELDS, coerce_list_param, add_filter_arguments
```

### Sequence utilities (`pe_common.sequence_utils`)

- `align_wt_mut_sequences()` — align wild-type and mutated sequences with padding
- `remove_padding()` — strip padding characters

### Data utilities (`pe_common.data_utils`)

- `build_test_mask_from_group_id()` — evaluation partition helpers

### Training (`pe_common.training`, lazy-loaded)

Requires PyTorch. Imported via `pe_common` namespace:

```python
from pe_common import (
    run_supervised_training_loop,
    fit_lightning_module,
    pearson_spearman,
    EarlyStopping,
    LightningTrainerConfig,
    build_lr_scheduler,
    resolve_training_seed,
    seed_training_run,
)
```

All three model wrappers share this module, so a change here affects DeepPrime,
PRIDICT2 and OPED at once. Notable behaviour:

- `seed_training_run(hyperparameters)` is called by each wrapper **before** the
  model is built, so weight init and data-loader shuffling are both seeded.
  `resolve_training_seed` returns the same value for `LightningTrainerConfig`.
- `fit_lightning_module` sets `enable_checkpointing=False` and returns the best
  state itself; callers decide where weights are persisted.
- `build_lr_scheduler` defaults `CosineAnnealingLR`'s `T_max` to the run's
  `max_epochs` rather than a fixed constant.

### Feature calculations (`pe_common.features`, lazy-loaded)

- `calculate_mfe()` — minimum free energy (ViennaRNA)
- `calculate_mt_wallace()` — melting temperature (Wallace method)
- `calculate_gc_content()` — GC content percentage

### Model interface (`pe_common.model_interface`)

Abstract `BasePEModel` contract implemented by Ensemble wrappers (`load_model`,
`prepare_data`, `predict`, `train`, `evaluate`, `save_model`).

## Design note

Standardized → model-format conversion lives in the **PE-DB** service
(`services/pe-db/app/formats/`, exposed via `GET /api/filter`).
`pe-common` intentionally stays free of model-specific conversion logic.

## Usage

```python
from pe_common import DATA_ROOT, DEVICE
from pe_common.devices import list_devices, resolve_device
from pe_common.sequence_utils import align_wt_mut_sequences
from pe_common.features import calculate_gc_content

print(f"Data directory: {DATA_ROOT}")
print(f"Default device: {DEVICE}")
print(f"Devices: {[d.device_id for d in list_devices()]}")

wt, mut = align_wt_mut_sequences("ATCG", "ATGCG", 2, 1, 1)
gc = calculate_gc_content("ATCGATCG")
```

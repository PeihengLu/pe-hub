# PE Common

Shared utilities package for PE Database and PE Ensemble services.

## Installation

```bash
pip install -e packages/pe-common
```

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
)
```

### Feature calculations (`pe_common.features`, lazy-loaded)

- `calculate_mfe()` — minimum free energy (ViennaRNA)
- `calculate_mt_wallace()` — melting temperature (Wallace method)
- `calculate_gc_content()` — GC content percentage

### Model interface (`pe_common.model_interface`)

Abstract `BasePEModel` contract implemented by Ensemble wrappers (`load_model`,
`prepare_data`, `predict`, `train`, `evaluate`, `save_model`).

## Design note

Standardized → model-format conversion lives in the **PE-DB** service
(`services/pe-db/app/utils/convert_data.py`, exposed via `GET /api/filter`).
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

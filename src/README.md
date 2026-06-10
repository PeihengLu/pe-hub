# Legacy Source Files

This directory previously held monolithic scripts and utilities. Most code has
been migrated into services and `packages/pe-common/`.

## Migration status

### Migrated to `packages/pe-common/`

- `constants.py` → `packages/pe-common/pe_common/constants.py`
- `sequence_utils.py` → `packages/pe-common/pe_common/sequence_utils.py`
- `features.py` → `packages/pe-common/pe_common/features.py`
- Device discovery → `packages/pe-common/pe_common/devices.py`
- Split assignment → `packages/pe-common/pe_common/splits.py`
- Training helpers → `packages/pe-common/pe_common/training.py`
- Model interface → `packages/pe-common/pe_common/model_interface.py`

### Migrated to `services/pe-db/`

- Data conversion → `app/converter.py`, `app/utils/standardize_data.py`, `app/utils/convert_data.py`
- Catalog metadata → `app/catalog/`

### Migrated to `services/pe-ensemble/`

- Model runners → `app/models/` (wrappers + `model_factory.py`)
- Training pipeline → `app/training/` (jobs, runner, device scheduler)
- CLI entry → `app/train_models.py`

### Remaining here

Any files still under `src/` are deprecated. New code should import from
`pe_common` or the relevant service package.

## Import guide

### Old (legacy)

```python
from src.constants import DATA_ROOT
from src.sequence_utils import align_wt_mut_sequences
```

### New (recommended)

```python
from pe_common import DATA_ROOT
from pe_common.sequence_utils import align_wt_mut_sequences
```

## When to remove

This directory can be deleted once no external scripts depend on it. Check with:

```bash
rg "from src\\." --glob '*.py'
```

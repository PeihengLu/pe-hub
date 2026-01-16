# Legacy Source Files

⚠️ **This directory contains legacy code that will be migrated to services.**

## Status

### ✅ Migrated to `packages/pe-common/`
- ~~`constants.py`~~ → `packages/pe-common/pe_common/constants.py`
- ~~`sequence_utils.py`~~ → `packages/pe-common/pe_common/sequence_utils.py`
- ~~`features.py`~~ → `packages/pe-common/pe_common/features.py`

### 🔄 Partially Migrated
- `data.py` - Some functions migrated to `services/pe-db/app/data_prep/converter.py`
  - Still contains additional utility functions not yet migrated

### 📦 To Be Migrated to `services/pe-ensemble/`
- `run_models.py` - Will move to `services/pe-ensemble/app/models/runner.py`
- `train_models.py` - Will move to `services/pe-ensemble/app/models/trainer.py`
- `utils.py` - Will move to `packages/pe-common/` or `services/pe-ensemble/`

## Migration Guide

To use the new structure, update your imports:

### Old (Legacy)
```python
from src.constants import DATA_ROOT
from src.sequence_utils import align_wt_mut_sequences
from src.features import calculate_gc_content
```

### New (Recommended)
```python
from pe_common import DATA_ROOT
from pe_common.sequence_utils import align_wt_mut_sequences
from pe_common.features import calculate_gc_content
```

## When to Remove

These files should be kept until:
1. PE Ensemble service is fully implemented
2. All data conversion logic is extracted and tested
3. All dependent scripts are updated
4. No external scripts depend on these files

See [MIGRATION.md](../MIGRATION.md) for detailed migration instructions.

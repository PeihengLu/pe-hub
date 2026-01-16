# PE Common

Shared utilities package for PE Database and PE Ensemble services.

## Installation

Install in development mode:

```bash
pip install -e packages/pe-common
```

## Contents

### Constants (`pe_common.constants`)
- `PROJECT_ROOT`: Root directory of the project
- `DATA_ROOT`: Path to datasets directory
- `MODEL_ROOT`: Path to vendor/models directory
- `DATABASE_ROOT`: Path to database directory
- `DEVICE`: Auto-detected PyTorch device (mps/cuda/cpu)

### Sequence Utilities (`pe_common.sequence_utils`)
- `align_wt_mut_sequences()`: Align wild-type and mutated sequences with padding
- `remove_padding()`: Remove padding characters from sequences

### Feature Calculations (`pe_common.features`)
- `calculate_mfe()`: Calculate Minimum Free Energy using ViennaRNA
- `calculate_mt_wallace()`: Calculate melting temperature using Wallace method
- `calculate_gc_content()`: Calculate GC content percentage

## Usage

```python
from pe_common import DATA_ROOT, DEVICE
from pe_common.sequence_utils import align_wt_mut_sequences
from pe_common.features import calculate_gc_content

# Use shared constants
print(f"Data directory: {DATA_ROOT}")
print(f"Using device: {DEVICE}")

# Use sequence utilities
wt, mut = align_wt_mut_sequences("ATCG", "ATGCG", 2, 1, 1)

# Calculate features
gc = calculate_gc_content("ATCGATCG")
```


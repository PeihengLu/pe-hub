# PE-Ensemble: Unified Model Interface

PE-Ensemble provides a unified interface for training and evaluating third-party Prime Editing prediction models.

## Architecture

```
services/pe-ensemble/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_factory.py          # Factory for creating model wrappers
│   │   ├── deepprime_wrapper.py      # DeepPrime model wrapper
│   │   ├── oped_wrapper.py           # OPED model wrapper (to be implemented)
│   │   ├── pridict_wrapper.py        # PRIDICT model wrapper (to be implemented)
│   │   └── pridict2_wrapper.py       # PRIDICT2 model wrapper (to be implemented)
│   └── main.py                        # FastAPI application
└── example_deepprime_usage.py         # Usage examples

packages/pe-common/
└── pe_common/
    └── model_interface.py              # Base interface for all models
```

## Quick Start

### 1. Install Dependencies

```bash
cd services/pe-ensemble
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from app.models.model_factory import ModelFactory
import torch

# Create a DeepPrime model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelFactory.create_model(
    'deepprime',
    device=device,
    pe_system='PE2max',
    cell_type='HEK293T'
)

# Load pre-trained model
model.load_model()

# Prepare your data
import pandas as pd
data = pd.read_csv('your_data.csv')
prepared_data = model.prepare_data(data)

# Make predictions
predictions = model.predict(prepared_data)

# Evaluate model
test_data = pd.read_csv('test_data_with_labels.csv')
metrics = model.evaluate(test_data)
print(f"Pearson: {metrics['pearson']:.4f}")
print(f"Spearman: {metrics['spearman']:.4f}")
```

## Supported Models

### DeepPrime

**Status**: ✅ Implemented

**Supported PE Systems**:
- PE2
- PE2max
- PE4max
- PE2max-e
- PE4max-e
- NRCH_PE2
- NRCH_PE2max
- NRCH_PE4max

**Supported Cell Types**:
- HEK293T
- A549
- DLD1
- HCT116
- HeLa
- MDA-MB-231
- NIH3T3

**Example**:
```python
model = ModelFactory.create_model(
    'deepprime',
    pe_system='PE2max',
    cell_type='HEK293T'
)
model.load_model()  # Uses default pre-trained model
```

### OPED

**Status**: ✅ Implemented

**Features**:
- Transformer-based architecture (Order 3)
- Training supported ✓
- Handles Target, PBS, RT sequences
- Attention-based sequence processing

**Example**:
```python
model = ModelFactory.create_model('oped')
model.load_model('/path/to/oped_model.pt')

# Or train from scratch
results = model.train(
    train_data=train_df,
    val_data=val_df,
    hyperparameters={'epoch_num': 100, 'lr': 0.001}
)
```

### PRIDICT2

**Status**: ✅ Implemented

**Supported Model Names**:
- `base_90k` - Trained on PRIDICT 1 schwank library
- `base_390k` - Multitask on schwank + hyongbum libraries
- `base_23k` - Trained on 23k library
- `base_90k_decinit_HEKschwank_FT` - Fine-tuned on 23k
- `base_390k_decinit_HEKhyongbum_FT` - Fine-tuned on 23k
- `base_390k_decinit_HEKschwank_FT` - Fine-tuned on 23k

**Supported Cell Types** (depends on model):
- HEK
- K562
- HEKschwank
- HEKhyongbum

**Features**:
- Predicts 3 outcomes: edited, unedited, indel
- RNN-based architecture
- Cell-type specific models

**Example**:
```python
model = ModelFactory.create_model(
    'pridict2', 
    wsize=20,
    model_name='base_390k'
)
model.load_model('/path/to/pridict2_model')

# Predict all three outcomes
prepared = model.prepare_data(df)
predictions = model.predict(prepared)
# Returns: [[edited, unedited, indel], ...]

# Or predict single outcome
edited_only = model.predict_single_outcome(prepared, 'averageedited')
```

## Model Interface

All model wrappers implement the `BasePEModel` interface:

### Core Methods

#### `load_model(model_path: str)`
Load a pre-trained model from disk.

```python
model.load_model('/path/to/model')
```

#### `prepare_data(df: pd.DataFrame, **kwargs) -> Any`
Convert input DataFrame to model-specific format.

```python
prepared_data = model.prepare_data(df)
```

#### `predict(data: Any, batch_size: int = 32) -> List[float]`
Make predictions on prepared data.

```python
predictions = model.predict(prepared_data, batch_size=128)
```

#### `train(train_data, val_data, hyperparameters) -> Dict`
Train the model (if training is supported).

```python
results = model.train(
    train_data=train_df,
    val_data=val_df,
    hyperparameters={'epochs': 100, 'lr': 0.001}
)
```

#### `evaluate(test_data: pd.DataFrame) -> Dict[str, float]`
Evaluate model on test data.

```python
metrics = model.evaluate(test_df)
# Returns: {'pearson': 0.85, 'spearman': 0.82, 'mse': 0.05, ...}
```

#### `save_model(model_path: str)`
Save trained model to disk.

```python
model.save_model('/path/to/save')
```

#### `get_model_info() -> Dict[str, Any]`
Get model metadata.

```python
info = model.get_model_info()
# Returns: {'name': 'DeepPrime', 'device': 'cuda', 'is_trained': True, ...}
```

## API Server

### Start the Server

```bash
cd services/pe-ensemble
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### API Endpoints

#### List Available Models
```bash
curl http://localhost:8001/models
```

Response:
```json
{
  "models": [
    {"name": "deepprime", "loaded": false}
  ],
  "count": 1
}
```

#### Load a Model
```bash
curl -X POST "http://localhost:8001/models/deepprime/load?model_path=/path/to/model"
```

#### Make Predictions
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deepprime",
    "data": [
      {"Target": "ACGT...", "PBS": "ACGT", ...},
      ...
    ],
    "batch_size": 128
  }'
```

#### Evaluate Model
```bash
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deepprime",
    "test_data": [...]
  }'
```

## Adding New Models

To add a new model wrapper:

### 1. Create Wrapper Class

Create `services/pe-ensemble/app/models/yourmodel_wrapper.py`:

```python
from pe_common.model_interface import BasePEModel
import pandas as pd
from typing import List, Dict, Any, Optional

class YourModelWrapper(BasePEModel):
    def __init__(self, device=None):
        super().__init__('YourModel', device)
    
    def load_model(self, model_path: str):
        # Load your model
        pass
    
    def prepare_data(self, df: pd.DataFrame, **kwargs):
        # Convert data to model format
        pass
    
    def predict(self, data, batch_size: int = 32) -> List[float]:
        # Make predictions
        pass
    
    def train(self, train_data, val_data=None, hyperparameters=None):
        # Train model (if supported)
        raise NotImplementedError()
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        # Evaluate model
        pass
    
    def save_model(self, model_path: str):
        # Save model
        pass
```

### 2. Register with Factory

Update `model_factory.py`:

```python
from .yourmodel_wrapper import YourModelWrapper

class ModelFactory:
    _models = {
        'deepprime': DeepPrimeModelWrapper,
        'yourmodel': YourModelWrapper,  # Add here
    }
```

### 3. Use Your Model

```python
model = ModelFactory.create_model('yourmodel')
model.load_model('/path/to/model')
predictions = model.predict(data)
```

## Data Format

### Input Data

Models expect standardized input format. See `datasets/standardized/` for examples.

Common columns:
- `Target`: Target sequence
- `PBS`: PBS sequence
- `RT`: RT template sequence
- `Edit_type`: Type of edit (substitution, insertion, deletion)
- `Edit_len`: Length of edit
- Cell type, PE system, and other metadata

### Output Format

Predictions are returned as:
```python
[0.45, 0.78, 0.32, ...]  # Predicted efficiencies
```

Evaluation metrics:
```python
{
    'pearson': 0.85,
    'spearman': 0.82,
    'mse': 0.05,
    'mae': 0.18,
    'n_samples': 1000
}
```

## Examples

See `example_deepprime_usage.py` for comprehensive examples:

```bash
cd services/pe-ensemble
python example_deepprime_usage.py
```

## Testing

```bash
pytest tests/test_model_wrappers.py
```

## Docker Deployment

```bash
cd services/pe-ensemble
docker build -t pe-ensemble .
docker run -p 8001:8001 pe-ensemble
```

## Contributing

1. Create a new branch
2. Implement your model wrapper following the `BasePEModel` interface
3. Add tests
4. Update documentation
5. Submit a pull request

## License

[Your License]

## Citation

If you use this work, please cite the original model papers:

- **DeepPrime**: [Citation]
- **OPED**: [Citation]
- **PRIDICT**: [Citation]
- **PRIDICT2**: [Citation]

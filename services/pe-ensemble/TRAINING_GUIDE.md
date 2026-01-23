# Training Pipeline Guide

This guide explains how to make the training pipeline functional for each model wrapper.

## Overview

Currently, most model wrappers have `NotImplementedError` in the `train()` method because training requires:

1. **Proper data preprocessing pipelines**
2. **Hyperparameter configuration**
3. **Training loops with optimization**
4. **Validation and checkpointing**
5. **Model architecture initialization**

## What You Need for Each Model

### 1. DeepPrime

**Status**: Training not supported (inference-only model)

**Requirements to enable training**:
- DeepPrime repository doesn't include training code
- Would need to implement from scratch or obtain from authors:
  - Custom CNN-GRU architecture training loop
  - Data augmentation strategies
  - Loss function (likely MSE with custom transformations)
  - Feature engineering pipeline

**Recommendation**: Use pre-trained models only. If custom training needed, contact DeepPrime authors.

---

### 2. OPED

**Status**: ✅ Training pipeline exists in vendor code

**What's needed**:

#### A. Data Requirements
```python
# Required columns in training data
required_columns = [
    'Target',      # Target sequence (as one-hot or encoded)
    'PBS',         # PBS sequence
    'RT',          # RT template sequence
    'Other',       # Additional features (20 features)
    'Efficiency'   # Target label (editing efficiency)
]
```

#### B. Implementation Steps

1. **Import training functions**:
```python
from vendor.models.oped.pegRNA_PredictingCodes.train_model import (
    train_and_test_transformer_order3,
    TransformerEncoderModelOrder3
)
```

2. **Prepare hyperparameters**:
```python
hyperparameters = {
    'ntoken': 4,                    # Number of tokens (A, C, G, T)
    'embedding_size': 64,           # Embedding dimension
    'hidden_size': [2048, 2048, 2048],  # Hidden layers for 3 transformers
    'hidden_size_fully': None,      # Fully connected layer sizes
    'output_size': 1,               # Regression output
    'nhead': 8,                     # Number of attention heads
    'num_encoder_layers': [6, 6, 6],  # Encoder layers for each transformer
    'drop_out': 0.1,
    'epoch_num': 100,
    'batch_size': 128,
    'lr': 0.001,
    'weight_decay': 0.0,
    'device': device,
    'best_epoch': True,
    'transfer': False,
    'freezing': False,
    'other_size': 0  # Size of "other" features
}
```

3. **The wrapper already implements this** - just ensure your data is properly formatted!

#### C. Data Preprocessing

OPED requires special preprocessing:
```python
from vendor.models.oped.pegRNA_PredictingCodes.read_data import read_data_of_ClinVar_file

# Your data needs these columns:
df = pd.DataFrame({
    'ID': [...],
    'Original': [...],      # Reference sequence
    'Substitution': [...],  # Edited sequence
    'Edit_type': [...],     # Type of edit
    # ... other required columns
})

prepared = read_data_of_ClinVar_file(df)
```

#### D. Training Command

```python
from app.models.model_factory import ModelFactory

# Create model
model = ModelFactory.create_model('oped')

# Train
results = model.train(
    train_data=train_df,
    val_data=val_df,
    hyperparameters={
        'epoch_num': 50,
        'batch_size': 256,
        'lr': 0.0005
    }
)

# Save
model.save_model('models/oped_trained.pt')
```

---

### 3. PRIDICT2

**Status**: ⚠️ Complex training pipeline exists but needs integration

**What's needed**:

#### A. Data Requirements

PRIDICT2 expects very specific data format:
```python
required_columns = [
    'seq_id',                      # Unique sequence ID
    'wide_initial_target',         # Initial target sequence (wide context)
    'wide_mutated_target',         # Mutated target sequence
    'Correction_Type',             # Type of correction
    'averageedited',               # Target: % edited
    'averageunedited',             # Target: % unedited
    'averageindel',                # Target: % indel
    # Plus many optional features...
]
```

#### B. Implementation Steps

The training is complex due to:
1. **Multiple model components**: Encoders, decoders, attention mechanisms
2. **Multi-task learning**: Predicts 3 outcomes simultaneously
3. **Cell-type specific decoders**
4. **Custom loss functions**: Balanced MSE loss

**To enable training**:

1. **Create training script** (`services/pe-ensemble/train_pridict2.py`):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'vendor' / 'models' / 'pridict2'))

from pridict.pridictv2.run_workflow import build_config_map, run_cont_pe_RNN_distribution
from pridict.pridictv2.hyperparam import RNNHyperparamConfig
from pridict.pridictv2.dataset import construct_load_multiple_dataloaders

# Define hyperparameters
trf_tup = (
    'LSTM',        # RNN class
    64,            # embed_dim
    2,             # num_hidden_layers
    True,          # bidirection
    0.3,           # p_dropout
    'relu',        # nonlin_func
    100,           # num_epochs
    128,           # batch_size
    0.0001         # l2_reg
)

experiment_options = {
    'model_name': 'PE_RNN_distribution',
    'experiment_desc': 'PRIDICT2 training',
    'annot_embed': 64,
    'assemb_opt': 'stack',
    'seqlevel_featdim': 50
}

# Build configuration
mconfig, options = build_config_map(trf_tup, experiment_options, loss_func='NPloss')

# Prepare data partition
data_partition = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

# Train
run_cont_pe_RNN_distribution(
    data_partition=data_partition,
    dsettypes=['train', 'val', 'test'],
    config=mconfig,
    options=options,
    wrk_dir='./models/pridict2',
    to_gpu=True
)
```

2. **Or integrate into wrapper**:

Update `pridict2_wrapper.py` train method:
```python
def train(self, train_data, val_data=None, hyperparameters=None):
    from pridict.pridictv2.run_workflow import build_config_map, run_cont_pe_RNN_distribution
    from pridict.pridictv2.hyperparam import RNNHyperparamConfig
    
    # Set default hyperparameters
    default_hp = {
        'rnn_class': 'LSTM',
        'embed_dim': 64,
        'num_hidden_layers': 2,
        'bidirection': True,
        'p_dropout': 0.3,
        'nonlin_func': 'relu',
        'num_epochs': 100,
        'batch_size': 128,
        'l2_reg': 0.0001
    }
    
    if hyperparameters:
        default_hp.update(hyperparameters)
    
    # Create tuple for config
    trf_tup = (
        default_hp['rnn_class'],
        default_hp['embed_dim'],
        default_hp['num_hidden_layers'],
        default_hp['bidirection'],
        default_hp['p_dropout'],
        default_hp['nonlin_func'],
        default_hp['num_epochs'],
        default_hp['batch_size'],
        default_hp['l2_reg']
    )
    
    # Build config
    experiment_options = {
        'model_name': 'PE_RNN_distribution',
        'experiment_desc': 'Training via wrapper',
        'annot_embed': 64,
        'assemb_opt': 'stack',
        'seqlevel_featdim': train_data.shape[1] - 3  # Adjust based on features
    }
    
    mconfig, options = build_config_map(trf_tup, experiment_options)
    
    # Prepare data partition
    data_partition = {
        'train': train_data,
        'val': val_data if val_data is not None else train_data,
    }
    
    # Train
    # ... (implement full training loop)
    
    return {'status': 'success'}
```

---

## General Steps to Enable Training

### Step 1: Prepare Training Data

Create a data preprocessing pipeline:

```python
# services/pe-ensemble/app/data/preprocessor.py

class DataPreprocessor:
    """Preprocess data for different model types"""
    
    @staticmethod
    def prepare_for_oped(df: pd.DataFrame) -> pd.DataFrame:
        """Convert standardized format to OPED format"""
        # Implement column mapping
        # Encode sequences
        # Add required features
        pass
    
    @staticmethod
    def prepare_for_pridict2(df: pd.DataFrame) -> pd.DataFrame:
        """Convert standardized format to PRIDICT2 format"""
        # Add wide context windows
        # Create correction type categories
        # Add position-specific features
        pass
```

### Step 2: Create Training Configuration

```python
# services/pe-ensemble/app/config/training_config.py

OPED_DEFAULT_CONFIG = {
    'embedding_size': 64,
    'hidden_size': [2048, 2048, 2048],
    'nhead': 8,
    'num_encoder_layers': [6, 6, 6],
    'dropout': 0.1,
    'epoch_num': 100,
    'batch_size': 128,
    'lr': 0.001,
    'weight_decay': 0.0
}

PRIDICT2_DEFAULT_CONFIG = {
    'rnn_class': 'LSTM',
    'embed_dim': 64,
    'num_hidden_layers': 2,
    'bidirection': True,
    'p_dropout': 0.3,
    'num_epochs': 100,
    'batch_size': 128,
    'l2_reg': 0.0001
}
```

### Step 3: Add Training Utilities

```python
# services/pe-ensemble/app/training/trainer.py

class ModelTrainer:
    """Unified training interface"""
    
    def __init__(self, model_wrapper, train_data, val_data):
        self.model = model_wrapper
        self.train_data = train_data
        self.val_data = val_data
    
    def train(self, hyperparameters=None, callbacks=None):
        """Train with callbacks for monitoring"""
        # Setup callbacks (TensorBoard, checkpointing, etc.)
        # Call model.train()
        # Log metrics
        pass
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save training checkpoint"""
        pass
    
    def load_checkpoint(self, path):
        """Resume from checkpoint"""
        pass
```

### Step 4: Add Experiment Tracking

```python
# Use MLflow or Weights & Biases

import mlflow

def train_with_tracking(model_name, train_data, val_data, hyperparameters):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(hyperparameters)
        
        # Create and train model
        model = ModelFactory.create_model(model_name)
        results = model.train(train_data, val_data, hyperparameters)
        
        # Log metrics
        mlflow.log_metrics(results)
        
        # Log model
        mlflow.pytorch.log_model(model.model, "model")
```

## Quick Start: Training OPED

The OPED model is the easiest to train. Here's a complete example:

```python
import pandas as pd
from app.models.model_factory import ModelFactory

# 1. Load your data (must have proper columns)
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')

# 2. Create model
model = ModelFactory.create_model('oped')

# 3. Train (the wrapper handles everything)
results = model.train(
    train_data=train_df,
    val_data=val_df,
    hyperparameters={
        'epoch_num': 50,
        'batch_size': 256,
        'lr': 0.0005,
        'weight_decay': 1e-5
    }
)

print(f"Training complete!")
print(f"Validation Pearson: {results['val_pearson']:.4f}")
print(f"Validation Spearman: {results['val_spearman']:.4f}")

# 4. Save model
model.save_model('models/oped_custom.pt')

# 5. Evaluate on test set
test_df = pd.read_csv('data/test.csv')
test_metrics = model.evaluate(test_df)
print(f"Test Pearson: {test_metrics['pearson']:.4f}")
```

## Testing Your Training Pipeline

Create tests:

```python
# tests/test_training.py

def test_oped_training():
    """Test OPED training with small dataset"""
    # Create small synthetic dataset
    train_data = create_synthetic_data(n=100)
    val_data = create_synthetic_data(n=20)
    
    model = ModelFactory.create_model('oped')
    
    results = model.train(
        train_data=train_data,
        val_data=val_data,
        hyperparameters={'epoch_num': 2}  # Quick test
    )
    
    assert results['status'] == 'success'
    assert 'val_pearson' in results
```

## Summary

**To make training functional**:

1. ✅ **OPED**: Already implemented - just need proper data format
2. ⚠️ **PRIDICT2**: Complex but possible - needs integration work
3. ❌ **DeepPrime**: Not available - use pre-trained models only

**Recommended approach**:
1. Start with OPED training (easiest)
2. Create data preprocessing utilities
3. Add experiment tracking
4. Integrate PRIDICT2 training if needed
5. Accept DeepPrime as inference-only

**Next steps**:
1. Create example training data in correct format
2. Test OPED training with small dataset
3. Add logging and monitoring
4. Document data format requirements
5. Create training scripts for each model

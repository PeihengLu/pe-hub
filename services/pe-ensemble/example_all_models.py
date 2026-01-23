"""
Complete example showing usage of all three model wrappers:
- DeepPrime
- OPED
- PRIDICT2
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'pe-common'))

from app.models.model_factory import ModelFactory


def example_all_models_comparison():
    """Compare all three models on the same data"""
    print("="*80)
    print("Example: Compare All Models")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # List all available models
    models = ModelFactory.list_models()
    print(f"Available models: {models}\n")
    
    # Model configurations
    model_configs = {
        'deepprime': {
            'kwargs': {
                'device': device,
                'pe_system': 'PE2max',
                'cell_type': 'HEK293T'
            },
            'model_path': None  # Uses default
        },
        'oped': {
            'kwargs': {'device': device},
            'model_path': 'path/to/oped_model.pt'  # Specify your model path
        },
        'pridict2': {
            'kwargs': {
                'device': device,
                'wsize': 20,
                'model_name': 'base_390k'
            },
            'model_path': 'path/to/pridict2_model'  # Specify your model path
        }
    }
    
    # Create and load each model
    loaded_models = {}
    for model_name, config in model_configs.items():
        print(f"--- {model_name.upper()} ---")
        try:
            # Create model
            model = ModelFactory.create_model(model_name, **config['kwargs'])
            
            # Load model
            if config['model_path']:
                model.load_model(config['model_path'])
            else:
                model.load_model()
            
            # Get model info
            info = model.get_model_info()
            print(f"✓ Loaded successfully")
            print(f"  Name: {info['name']}")
            print(f"  Device: {info['device']}")
            print(f"  Trained: {info['is_trained']}")
            
            # Model-specific info
            if model_name == 'deepprime':
                print(f"  PE System: {info['pe_system']}")
                print(f"  Cell Type: {info['cell_type']}")
                print(f"  Ensemble Size: {info['n_models']}")
            elif model_name == 'pridict2':
                print(f"  Model Name: {info['model_name']}")
                print(f"  Outcomes: {', '.join(info['outcomes'])}")
            
            loaded_models[model_name] = model
            
        except FileNotFoundError as e:
            print(f"✗ Model files not found: {e}")
        except Exception as e:
            print(f"✗ Error loading: {e}")
        
        print()
    
    return loaded_models


def example_deepprime_detailed():
    """Detailed DeepPrime example"""
    print("="*80)
    print("DeepPrime: Detailed Example")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different PE systems
    pe_systems_to_test = [
        ('PE2max', 'HEK293T'),
        ('PE4max', 'A549'),
        ('NRCH_PE2max', 'HEK293T'),
    ]
    
    for pe_system, cell_type in pe_systems_to_test:
        print(f"\n--- {pe_system} in {cell_type} ---")
        try:
            model = ModelFactory.create_model(
                'deepprime',
                device=device,
                pe_system=pe_system,
                cell_type=cell_type
            )
            model.load_model()
            
            info = model.get_model_info()
            print(f"✓ Model loaded: {info['n_models']} ensemble models")
            print(f"  Model type: {info['model_type']}")
            
        except Exception as e:
            print(f"✗ {e}")


def example_oped_training():
    """Example: Train OPED model"""
    print("="*80)
    print("OPED: Training Example")
    print("="*80)
    
    print("\nNote: This example shows how to train OPED.")
    print("You need to provide properly formatted training data.\n")
    
    # Example training workflow
    example_code = """
# Load your training data
train_df = pd.read_csv('data/oped_train.csv')
val_df = pd.read_csv('data/oped_val.csv')

# Required columns in data:
# - Target, PBS, RT sequences (encoded)
# - Other features (20 additional features)
# - Efficiency (target label)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ModelFactory.create_model('oped', device=device)

# Train
results = model.train(
    train_data=train_df,
    val_data=val_df,
    hyperparameters={
        'epoch_num': 50,
        'batch_size': 256,
        'lr': 0.0005,
        'weight_decay': 1e-5,
        'embedding_size': 64,
        'hidden_size': [2048, 2048, 2048],
        'nhead': 8,
        'num_encoder_layers': [6, 6, 6],
        'drop_out': 0.1
    }
)

print(f"Training completed!")
print(f"Validation Pearson: {results['val_pearson']:.4f}")
print(f"Validation Spearman: {results['val_spearman']:.4f}")

# Save trained model
model.save_model('models/oped_custom.pt')

# Evaluate on test set
test_df = pd.read_csv('data/oped_test.csv')
metrics = model.evaluate(test_df)
print(f"Test Pearson: {metrics['pearson']:.4f}")
print(f"Test Spearman: {metrics['spearman']:.4f}")
"""
    
    print(example_code)


def example_pridict2_outcomes():
    """Example: PRIDICT2 multi-outcome prediction"""
    print("="*80)
    print("PRIDICT2: Multi-Outcome Prediction")
    print("="*80)
    
    print("\nPRIDICT2 predicts three outcomes simultaneously:")
    print("  1. averageedited - % of edited outcomes")
    print("  2. averageunedited - % of unedited outcomes")
    print("  3. averageindel - % of indel outcomes\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Show supported models
    supported = PRIDICT2ModelWrapper.get_supported_models()
    print("Supported model configurations:")
    for model_name, cell_types in supported.items():
        print(f"  {model_name}: {', '.join(cell_types)}")
    
    print("\nExample usage:")
    example_code = """
# Create PRIDICT2 model
model = ModelFactory.create_model(
    'pridict2',
    device=device,
    wsize=20,
    model_name='base_390k'
)

# Load pre-trained model
model.load_model('/path/to/trained_models/base_390k')

# Prepare data
df = pd.read_csv('your_data.csv')
prepared = model.prepare_data(
    df,
    cell_types=['HEK'] * len(df),  # Specify cell type for each sample
    batch_size=500
)

# Predict all three outcomes
predictions = model.predict(prepared)
# Returns: [[edited, unedited, indel], ...]

# Or predict just one outcome
edited = model.predict_single_outcome(prepared, 'averageedited')

# Evaluate if you have ground truth
test_df = pd.read_csv('test_with_labels.csv')
metrics = model.evaluate(test_df)
print(f"Edited - Pearson: {metrics['averageedited_pearson']:.4f}")
print(f"Unedited - Pearson: {metrics['averageunedited_pearson']:.4f}")
print(f"Indel - Pearson: {metrics['averageindel_pearson']:.4f}")
"""
    
    print(example_code)


def example_model_comparison_workflow():
    """Example: Compare predictions from all models"""
    print("="*80)
    print("Workflow: Compare Model Predictions")
    print("="*80)
    
    example_code = """
import pandas as pd
from app.models.model_factory import ModelFactory

# Load test data (standardized format)
test_df = pd.read_csv('data/test_standardized.csv')

# Initialize models
models = {
    'deepprime': ModelFactory.create_model('deepprime', pe_system='PE2max', cell_type='HEK293T'),
    'oped': ModelFactory.create_model('oped'),
    'pridict2': ModelFactory.create_model('pridict2', model_name='base_390k')
}

# Load models
models['deepprime'].load_model()
models['oped'].load_model('models/oped.pt')
models['pridict2'].load_model('models/pridict2/base_390k')

# Prepare data for each model
predictions = {}
for name, model in models.items():
    print(f"Predicting with {name}...")
    
    # Prepare data (each model has its own format)
    prepared = model.prepare_data(test_df)
    
    # Make predictions
    preds = model.predict(prepared)
    
    # For PRIDICT2, extract edited outcome
    if name == 'pridict2':
        preds = [p[0] for p in preds]  # Get 'averageedited' only
    
    predictions[name] = preds

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'true_efficiency': test_df['Efficiency'],
    'deepprime_pred': predictions['deepprime'],
    'oped_pred': predictions['oped'],
    'pridict2_pred': predictions['pridict2']
})

# Calculate correlations
from scipy.stats import pearsonr

for model_name in ['deepprime', 'oped', 'pridict2']:
    r, p = pearsonr(
        comparison_df['true_efficiency'],
        comparison_df[f'{model_name}_pred']
    )
    print(f"{model_name}: r = {r:.4f}, p = {p:.4e}")

# Ensemble prediction (average)
comparison_df['ensemble_pred'] = comparison_df[
    ['deepprime_pred', 'oped_pred', 'pridict2_pred']
].mean(axis=1)

r_ensemble, _ = pearsonr(
    comparison_df['true_efficiency'],
    comparison_df['ensemble_pred']
)
print(f"\\nEnsemble: r = {r_ensemble:.4f}")

# Save results
comparison_df.to_csv('results/model_comparison.csv', index=False)
"""
    
    print(example_code)


def main():
    print("\n" + "="*80)
    print("PE-Ensemble: Complete Model Wrappers Demo")
    print("="*80 + "\n")
    
    # Import here to avoid errors if not all dependencies available
    try:
        from app.models.pridict2_wrapper import PRIDICT2ModelWrapper
    except:
        PRIDICT2ModelWrapper = None
    
    # Run examples
    try:
        example_all_models_comparison()
    except Exception as e:
        print(f"Error in comparison: {e}\n")
    
    example_deepprime_detailed()
    example_oped_training()
    
    if PRIDICT2ModelWrapper:
        example_pridict2_outcomes()
    
    example_model_comparison_workflow()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nFor more details, see:")
    print("  - README_MODELS.md - Complete API documentation")
    print("  - TRAINING_GUIDE.md - How to enable training pipelines")
    print("  - tests/test_model_wrappers.py - Unit tests")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

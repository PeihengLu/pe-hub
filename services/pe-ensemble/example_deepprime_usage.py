"""
Example usage of DeepPrime model wrapper in PE-Ensemble

This script demonstrates how to:
1. Load a DeepPrime model
2. Make predictions
3. Evaluate model performance
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'packages' / 'pe-common'))

from app.models.model_factory import ModelFactory


def example_load_and_predict():
    """Example: Load DeepPrime model and make predictions"""
    print("="*60)
    print("Example 1: Load DeepPrime and Make Predictions")
    print("="*60)
    
    # Create DeepPrime model for PE2max in HEK293T cells
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create_model(
        'deepprime',
        device=device,
        pe_system='PE2max',
        cell_type='HEK293T'
    )
    
    # Load pre-trained model (uses default from repository)
    print("\nLoading DeepPrime model...")
    model.load_model()
    
    # Print model info
    print("\nModel Info:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Example input data (you would load your actual data)
    # Note: This is a placeholder - real data needs proper feature columns
    sample_data = pd.DataFrame({
        # Add required DeepPrime input columns here
        # See DeepPrime documentation for required format
    })
    
    # If you have actual data:
    # sample_data = pd.read_csv('your_input_data.csv')
    
    if not sample_data.empty:
        # Prepare data
        print("\nPreparing data...")
        prepared_data = model.prepare_data(sample_data)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(prepared_data)
        
        print(f"\nPredictions (first 10): {predictions[:10]}")
    else:
        print("\n[Note] No sample data provided. Skipping prediction step.")


def example_evaluate():
    """Example: Evaluate DeepPrime model on test data"""
    print("\n" + "="*60)
    print("Example 2: Evaluate DeepPrime Model")
    print("="*60)
    
    # Create and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelFactory.create_model(
        'deepprime',
        device=device,
        pe_system='PE2max',
        cell_type='HEK293T'
    )
    
    print("\nLoading model...")
    model.load_model()
    
    # Load test data with efficiency labels
    # test_data = pd.read_csv('test_data_with_labels.csv')
    
    # For demonstration, create dummy data
    test_data = pd.DataFrame({
        # Add required columns + 'Efficiency' column
    })
    
    if not test_data.empty and 'Efficiency' in test_data.columns:
        print("\nEvaluating model...")
        metrics = model.evaluate(test_data)
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("\n[Note] No test data with labels provided. Skipping evaluation.")


def example_different_pe_systems():
    """Example: Use different PE systems and cell types"""
    print("\n" + "="*60)
    print("Example 3: Different PE Systems and Cell Types")
    print("="*60)
    
    configs = [
        ('PE2max', 'HEK293T'),
        ('PE4max', 'A549'),
        ('NRCH_PE2max', 'HEK293T'),
    ]
    
    for pe_system, cell_type in configs:
        print(f"\n--- {pe_system} in {cell_type} ---")
        
        try:
            model = ModelFactory.create_model(
                'deepprime',
                pe_system=pe_system,
                cell_type=cell_type
            )
            
            # Check if this combination is available
            model.load_model()
            print(f"✓ Model loaded successfully")
            print(f"  Ensemble size: {model.get_model_info()['n_models']} models")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def example_list_available_models():
    """Example: List all available models"""
    print("\n" + "="*60)
    print("Example 4: List Available Models")
    print("="*60)
    
    models = ModelFactory.list_models()
    print(f"\nAvailable models: {models}")
    
    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")
        info = ModelFactory.get_model_info(model_name)
        for key, value in info.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")


def example_batch_prediction():
    """Example: Batch prediction on multiple samples"""
    print("\n" + "="*60)
    print("Example 5: Batch Prediction")
    print("="*60)
    
    # Load model
    model = ModelFactory.create_model(
        'deepprime',
        pe_system='PE2max',
        cell_type='HEK293T'
    )
    model.load_model()
    
    # Load standardized data from your database
    # For example, from datasets/standardized/deepprime/
    data_path = Path(__file__).parent / 'datasets' / 'standardized' / 'deepprime'
    
    if data_path.exists():
        csv_files = list(data_path.glob('*.csv'))
        if csv_files:
            print(f"\nFound {len(csv_files)} data files")
            
            # Load first file as example
            sample_file = csv_files[0]
            print(f"\nLoading: {sample_file.name}")
            
            df = pd.read_csv(sample_file)
            print(f"  Samples: {len(df)}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
            # If data has required format, make predictions
            # prepared_data = model.prepare_data(df)
            # predictions = model.predict(prepared_data)
            # df['predicted_efficiency'] = predictions
            # print(f"\n  Predictions completed!")
        else:
            print("\nNo CSV files found in standardized/deepprime/")
    else:
        print(f"\n[Note] Data directory not found: {data_path}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DeepPrime Model Wrapper - Usage Examples")
    print("="*60)
    
    # Run examples
    try:
        example_list_available_models()
        example_different_pe_systems()
        example_load_and_predict()
        # example_evaluate()
        # example_batch_prediction()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

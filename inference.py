"""Inference using existing models on existing data.
"""
from typing import List

from pe_common import DEVICE, MODEL_ROOT, DATA_ROOT
from src.run_models import run_oped
from datasets.data import std_to_oped

if __name__ == "__main__":
    # Example usage
    model = "dp"
    cell_line = "HEK293T"
    pe_system = "pe2"
    
    # Convert standardized data to OPED format
    oped_data = std_to_oped(
        model=model,
        cell_line=cell_line,
        pe_system=pe_system
    )
    
    # Run OPED model inference
    predictions, weights = run_oped(
        data=oped_data
    )
    
    print(f"Predictions for {model} on {cell_line} using {pe_system}: {predictions}")
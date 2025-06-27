from typing import List
import sys

from models.DeepPrime.models.load_model import load_deepprime, load_deepspcas9
import pandas as pd

from src.utils import get_model_path, get_data_path

def run_pridict(data_path: str) -> List[float]:
    """
    Run the prediction model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    # Placeholder for actual prediction logic
    return [0.0, 1.0, 2.0]  # Example output

def run_pridict2(data_path: str) -> List[float]:
    """
    Run the second prediction model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    # Placeholder for actual prediction logic
    return [3.0, 4.0, 5.0]  # Example output

def run_oped(data_path: str) -> List[float]:
    """
    Run the OPED model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    # Placeholder for actual prediction logic
    return [6.0, 7.0, 8.0]  # Example output

def run_deepprime(data_path: str) -> List[float]:
    """
    Run the DeepPrime model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    from models.DeepPrime.src.dprime import calculate_deepprime_score
    
    df = pd.read_csv(data_path)
    calculate_deepprime_score(df, pe_system='PE2max', cell_type='HEK293T')
    
    # Placeholder for actual prediction logic
    return [9.0, 10.0, 11.0]  # Example output

def run_spcas9(data_path: str) -> List[float]:
    """
    Run the DeepSpCas9 model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    # Placeholder for actual prediction logic
    return [12.0, 13.0, 14.0]  # Example output
from typing import List
import sys

from models.DeepPrime.models.load_model import load_deepprime, load_deepspcas9
from models.PRIDICT.prieml.predict_outcomedistrib import PRIEML_Model
import pandas as pd

from src.constants import DATA_ROOT, MODEL_ROOT, DEVICE

def run_pridict(data_path: str) -> List[float]:
    """
    Run the prediction model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    # trained model directory
    model_dir = (
        MODEL_ROOT / 'PRIDICT' / 'trained_models' /
        'schwank_rnnattn' / 'v3' / 'train_val')

    device = DEVICE
    prieml_model = PRIEML_Model(device,wsize=20, normalize='max', fdtype=torch.float32)

    tcol = 'averageedited'
    res_lst = []
    for wsize in [20]:
        prieml_model.wsize = wsize
        dloader = prieml_model.prepare_data(df_test, y_ref=[tcol], batch_size=1500)
        pred_lst=[]
        for run in range(5):
            # predict
            mdir = os.path.join(model_dir, f'run_{run}')
            pred_df = prieml_model.predict_from_dloader(dloader, mdir, y_ref=[tcol])
            pred_lst.append(pred_df)
            pear_score = compute_pearson_corr(pred_df[f'true_{tcol}'],pred_df[f'pred_{tcol}'])[0]
            spear_score = compute_spearman_corr(pred_df[f'true_{tcol}'], pred_df[f'pred_{tcol}'])[0]
            print('pearson corr:', pear_score)
            print('spearman corr:',spear_score)
            res_lst.append((wsize, run, pear_score, spear_score))
            print('-'*15)
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
    # TODO: make sure the data is in the correct format
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
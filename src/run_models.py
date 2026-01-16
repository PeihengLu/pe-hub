# src/run_models.py
# -*- coding: utf-8 -*-
"""
This module provides functions to evaluate various models 
using their pretrained weights
"""
from typing import List
import sys
import pathlib
import torch
import pandas as pd
import os
from os.path import join  as pjoin
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from datasets.data import load_data
from src.constants import DATA_ROOT, MODEL_ROOT, DEVICE

def run_pridict(data_path: str, distribution: bool = False) -> List[float]:
    """
    Run the prediction model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    from models.pridict.prieml.predict_outcomedistrib import PRIEML_Model
    device = DEVICE
    # trained model directory
    model_dir = pjoin(MODEL_ROOT, 'pridict',
                             'trained_models',
                             'schwank_rnnattn',
                             'v3',
                             'train_val')
    prieml_model = PRIEML_Model(device, wsize=20, normalize='max', fdtype=torch.float32)

    tcol = ['averageedited'] if not distribution else ['averageedited', 'averageunedited', 'averagebystander']
    # TODO: refactor to using tcol as a list
    res_lst = []
    for wsize in [20]:
        prieml_model.wsize = wsize
        dloader = prieml_model.prepare_data(df_test, y_ref=tcol, batch_size=1500)
        pred_lst = []
        for run in range(5):
            # predict
            mdir = os.path.join(model_dir, f'run_{run}')
            pred_df = prieml_model.predict_from_dloader(dloader, mdir, y_ref=tcol)
            pred_lst.append(pred_df)
            pear_score = compute_pearson_corr(pred_df[f'true_{tcol}'], pred_df[f'pred_{tcol}'])[0]
            spear_score = compute_spearman_corr(pred_df[f'true_{tcol}'], pred_df[f'pred_{tcol}'])[0]
            print('pearson corr:', pear_score)
            print('spearman corr:', spear_score)
            res_lst.append((wsize, run, pear_score, spear_score))
            print('-' * 15)

        pred_df = pd.concat(pred_lst, axis=0, ignore_index=True)
        pred_df = prieml_model.compute_avg_predictions(pred_df)
        print('wsize:', wsize)

        pear_score = compute_pearson_corr(pred_df[f'true_{tcol}'], pred_df[f'pred_{tcol}'])[0]
        spear_score = compute_spearman_corr(pred_df[f'true_{tcol}'], pred_df[f'pred_{tcol}'])[0]
        print('pearson corr:', pear_score)
        print('spearman corr:', spear_score)
        res_lst.append((wsize, 'avg_runs', pear_score, spear_score))
        print('-' * 15)

    res_df = pd.DataFrame(res_lst)
    res_df.columns = ['wsize', 'run', 'pear_score', 'spearman_score']

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

def run_oped(
        pe_system: str = '',
        cell_line: str = '',
        model: str = '',
        data: pd.DataFrame = None,
    ) -> List[float]:
    """
    Run the OPED model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    from models.oped.pegRNA_PredictingCodes.predict_efficiency_of_ClinVar import read_data_of_Single
    from models.oped.pegRNA_PredictingCodes.evaluate_model import transformer_predictor_order3
    from models.oped.pegRNA_PredictingCodes.train_model import load_model as load_oped_model
    
    if data is None:
        data = load_data(
            cell_line=cell_line,
            pe_system=pe_system,
            src_model=model,
            target_model='oped',
        )

        if not data:
            raise ValueError(f"No data found for pe_system={pe_system}, cell_line={cell_line}, model={model}")
    
    # 'Target(47bp)', 'PBS', 'RT'
    if 'Target(47bp)' not in data.columns or \
       'PBS' not in data.columns or \
       'RT' not in data.columns:
        raise ValueError("Data must contain 'Target(47bp)', 'PBS', and 'RT' columns.")
    
    # prepare the data for the model
    data = read_data_of_Single(data)
    
    print(MODEL_ROOT / 'oped' / 'pegRNA_PredictingCodes' / 'Model_Trained')
    transformer = load_oped_model(
        DEVICE, model_dir=(
            MODEL_ROOT / 'oped' / 'pegRNA_PredictingCodes' / 'Model_Trained'),
        model_name='pegRNA_Model_Merged_saved.order3_decoder.pt'
    )  # load model

    efficiency, attention_weights = transformer_predictor_order3(transformer, data, 1024, DEVICE)
    print(efficiency)
    print(attention_weights)
    
    return efficiency # Example output

def run_deepprime(data_path: str) -> List[float]:
    """
    Run the DeepPrime model on the given data path.
    
    Args:
        data_path (str): The path to the data file.
        
    Returns:
        List[float]: A list of predictions.
    """
    from models.deepprime.models.load_model import load_deepprime, load_deepspcas9
    from models.deepprime.src.dprime import calculate_deepprime_score
    
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
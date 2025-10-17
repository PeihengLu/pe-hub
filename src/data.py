# src/data.py
# -*- coding: utf-8 -*-
"""Data conversion script for the project.
This script contains functions to convert data from one format to another.
"""
import os
import ast
import sys
from pathlib import Path
import functools
sys.path.insert(0, str(Path(__file__).parent.parent)) 

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.constants import DATA_ROOT
from src.sequence_utils import align_wt_mut_sequences, remove_padding

endogenetic_data = 'endogenetic'
library_study = 'library-study'

# ==============================================================================
# Source data manipulation functions
# fast recovery of the original data
# ==============================================================================

def read_from_deepprime_org(original_data_path: str, sheet_name: str = '1') -> None:
    """
    Read from the deepprime original excel file
    Args:   
        original_data_path (str): Path to the original deep prime data.
        output_path (str): Path to save the converted data.
    """
    # Load the original data, skipping the first 3 rows and use the forth row as header
    original_data = pd.read_excel(original_data_path, sheet_name=sheet_name, skiprows=3, header=0)
    
    # Rename columns to remove spaces and special characters(newlines, tabs, etc.)
    original_data.columns = original_data.columns.str.replace(' ', '_').str.replace('\n', '').str.replace('\t', '')
    
    # change the rest of the columns to lower case
    original_data.columns = original_data.columns.str.lower()
    
    # Rename some more opaque columns
    original_data.rename(columns={
        "wide_target_sequence(target_74bps_=_4bp_neighboring_sequence_+_20_bp_protospacer_+_3_bp_ngg_+_47_bp_neighboring_sequence)": "wt-sequence",
        "edited_target_sequence(target_74bps_=_rt-pbs_corresponding_region_and_masked_by_'x')": "mut-sequence",
    }, inplace=True)
    
    return original_data

def export_deepprime_all() -> None:
    """
    Read the original deep prime data to a more usable csv format.
    Args:
        original_data_path (str): Path to the original deep prime data.
        output_path (str): Path to save the converted data.
    """
    # Read the first sheet and use it as catalog
    DP_ORG_DATA_PATH = DATA_ROOT / library_study / 'deepprime-org' / 'deepprime-org.xlsx'
    original_data_catalog = pd.read_excel(
        DP_ORG_DATA_PATH, sheet_name='Summary', header=0,
    )
    # Rename the Index Column to 'SheetName'
    original_data_catalog.rename(columns={'Index': 'Sheet name'}, inplace=True)
    
    print(original_data_catalog.head())
    
    dp_model_to_datasource = {
        'DeepPrime': 'dp',
        'DeepPrime-FT': 'dp_ft',
    }
        
    for index, row in original_data_catalog.iterrows():
        # Get the sheet name, cell line, pe system and model
        sheet_name = row['Sheet name']
        cell_line = row['Cell line']
        pe_system = row['PE system']
        model = row['Model']
        
        if model not in dp_model_to_datasource:
            print(f"Model {model} is not supported. Skipping.")
            continue
    
        # Load the original data, skipping the first 3 rows and use the forth row as header
        original_data = read_from_deepprime_org(
            DP_ORG_DATA_PATH, sheet_name=sheet_name
        )
        
        # save the data to a csv file
        output_path = DATA_ROOT / library_study / 'deepprime' / f"dp-{dp_model_to_datasource[model]}-{cell_line.lower()}-{pe_system.lower()}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_data.to_csv(output_path, index=False)
        
def export_pridict2_all(original_data_path: str) -> None:
    """
    Split pridict 2 data into four parts, one for each cell line
    """
    # Load the original data, skipping the first 3 rows and use the forth row as header
    original_data = pd.read_csv(original_data_path, sep='\t', header=0)
    
    # look for the four cell lines
    cell_lines = [
        'HEK',
        'K562',
        'K562MLH1dn',
        'AdV'
    ]


# =============================================================================
# Data format conversion
# =============================================================================

def deepprime_org_to_std(data: pd.DataFrame, cell_line: str, pe_system: str, model: str = None) -> None:
    '''
    Convert the data from DeepPrime format to the standard format
    '''
    if not model:
        target = f"std-dp-{cell_line}-{pe_system}.csv"
    else:
        target = f"std-{model}-{cell_line}-{pe_system}.csv"
        
    # if isfile(pjoin('../', 'std', target)):
    #     return

    # replace the '-' in editor and cell line with '_'
    cell_line = cell_line.lower()
    pe_system = pe_system.lower()
    cell_line = cell_line.replace('-', '_')
    pe_system = pe_system.replace('-', '_')

    output = []

    # result columns
    result_columns = [
        'cell-line', 'group-id', 'mut-type', 'edit-len', 'wt-sequence', 'mut-sequence', 
        'protospacer-location-l', 'protospacer-location-r', 
        'pbs-location-l', 'pbs-location-r', 'rtt-location-l', 'rtt-location-r', 
        'lha-location-l', 'lha-location-r', 'rha-location-l', 'rha-location-r', 
        'spcas9-score', 'editing-efficiency', 'fold']

    g_id = 0
    prev = ""

    # iterate over the data
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        wt_sequence = item['wt-sequence']
        mut_sequence = item['mut-sequence']

        group_id = ind
        edit_len = item['Edit_len']
        rha_len = item['RHA_len']
        pbs_rtt_len = item['RT-PBSlen']

        # edit type
        if item['type_sub']:
            mut_type = 0
        elif item['type_ins']:
            mut_type = 1
        elif item['type_del']:
            mut_type = 2
        
        protospacer_location_l = 5
        protospacer_location_r = 24

        protospacer = wt_sequence[protospacer_location_l:protospacer_location_r]
        
        # assign group based on target loci
        if prev:
            if protospacer == prev:
                group_id = g_id
            else:
                g_id += 1
                group_id = g_id
        else:
            group_id = g_id
        prev = protospacer

        pbs_len = item['PBSlen']
        rtt_len = item['RTlen']

        pbs_location_l = -1
        for ind, c in enumerate(mut_sequence):
            if c != 'x':
                pbs_location_l = ind
                break
        pbs_location_r = pbs_location_l + pbs_len
        lha_location_l = pbs_location_r
        if mut_type == 2: # deletion
            lha_location_r = pbs_location_r + (pbs_rtt_len - pbs_len - rha_len)
        else:
            lha_location_r = pbs_location_r + (pbs_rtt_len - pbs_len - rha_len - edit_len)

        if mut_type == 2: # deletion
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len + edit_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len + edit_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len
        elif mut_type == 1: # insertion
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len - edit_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len - edit_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len
        else: # length does not change
            rha_location_wt_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_wt_r = pbs_location_l + pbs_rtt_len
            rha_location_mut_l = pbs_location_l + pbs_rtt_len - rha_len
            rha_location_mut_r = pbs_location_l + pbs_rtt_len 

        rtt_location_wt_l = pbs_location_r
        rtt_location_wt_r = rha_location_wt_r
        rtt_location_mut_l = pbs_location_r
        rtt_location_mut_r = rha_location_mut_r
        
        rtt_location_l = rtt_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rtt_location_r = rtt_location_wt_r
        else:
            rtt_location_r = rtt_location_mut_r
            
        rha_location_l = rha_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rha_location_r = rha_location_wt_r
        else:
            rha_location_r = rha_location_mut_r

        # remove the mask of the mutated sequence
        mut_sequence = ''
        mut_sequence += wt_sequence[:lha_location_r]
        mut_sequence += mut_sequence[lha_location_r:rha_location_mut_r]
        if mut_type == 1: # insertion
            mut_sequence += wt_sequence[rha_location_wt_r:len(wt_sequence)-edit_len]
        else:
            mut_sequence += wt_sequence[rha_location_wt_r:]
            
        wt_sequence, mut_sequence = align_wt_mut_sequences(
            wt_sequence, mut_sequence, lha_location_r, 
            edit_length=edit_len, edit_type=mut_type)
        
        spcas9_score = item['DeepSpCas9_score']
        editing_efficiency = item['Measured_PE_efficiency']
        fold = item['fold']
        
        # # pad the mutated sequence to the same length as the wildtype sequence
        # if len(mut_sequence) < len(wt_sequence):
        #     mut_sequence += 'N' * (len(wt_sequence) - len(mut_sequence))
        
        output.append([
            cell_line, group_id, mut_type, edit_len, wt_sequence, mut_sequence, 
            protospacer_location_l, protospacer_location_r, 
            pbs_location_l, pbs_location_r, 
            rtt_location_l, rtt_location_r, 
            lha_location_l, lha_location_r, 
            rha_location_l, rha_location_r, 
            spcas9_score, editing_efficiency, fold])

    # save the extracted information
    output_df = pd.DataFrame(output, columns=result_columns, index=None)
    # convert the columns to the correct types
    output_df['wt-sequence'] = output_df['wt-sequence'].str.upper()
    output_df['mut-sequence'] = output_df['mut-sequence'].str.upper()
    output_df['cell-line'] = output_df['cell-line'].str.lower()
    output_df['group-id'] = output_df['group-id'].astype(int)
    output_df['mut-type'] = output_df['mut-type'].astype(int)
    output_df['edit-len'] = output_df['edit-len'].astype(int)
    output_df['protospacer-location-l'] = output_df['protospacer-location-l'].astype(int)
    output_df['protospacer-location-r'] = output_df['protospacer-location-r'].astype(int) 
    output_df['pbs-location-l'] = output_df['pbs-location-l'].astype(int)
    output_df['pbs-location-r'] = output_df['pbs-location-r'].astype(int)
    output_df['rtt-location-l'] = output_df['rtt-location-l'].astype(int)
    output_df['rtt-location-r'] = output_df['rtt-location-r'].astype(int)
    output_df['lha-location-l'] = output_df['lha-location-l'].astype(int)
    output_df['lha-location-r'] = output_df['lha-location-r'].astype(int)
    output_df['rha-location-l'] = output_df['rha-location-l'].astype(int)
    output_df['rha-location-r'] = output_df['rha-location-r'].astype(int)
    output_df['spcas9-score'] = output_df['spcas9-score'].astype(float)
    output_df['editing-efficiency'] = output_df['editing-efficiency'].astype(float)
    output_df['fold'] = output_df['fold'].astype(str)
    # export the data to a csv file
    output_path = DATA_ROOT / 'std' / target
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False) 

    
def pridict2_to_std(data: pd.DataFrame, cell_line: str, pe_system: str, model: str = 'pd2') -> pd.DataFrame:
    '''
    Convert the data from PRIDICT2 format to the standard format
    '''
    output = []
    
    # drop the rows where wt-sequence or mut-sequence is empty
    data = data.dropna(subset=['wide_initial_target', 'wide_mutated_target'])

    # cell lines
    cell_lines = {'HEKaverageedited': 'hek293t','K562averageedited': 'k562','K562MLH1dnaverageedited': 'k562mlh1dn','AdVaverageedited': 'adv'}

    # result columns
    result_columns = ['cell-line', 'group-id', 'mut-type', 'edit-len', 'wt-sequence', 'mut-sequence', 'protospacer-location-l', 'protospacer-location-r', 'pbs-location-l', 'pbs-location-r', 'rtt-location-l', 'rtt-location-r', 'lha-location-l', 'lha-location-r', 'rha-location-l', 'rha-location-r', 'spcas9-score', 'editing-efficiency']

    # enum of mutation types
    mutation_types = ['1bpReplacement', 'MultibpReplacement', 'Insertion', 'Deletion']
    
    group_prev = -1
    group_id = -1

    # extract the important information
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        if item['group'] != group_prev:
            group_id += 1
            group_prev = item['group']
        wt_sequence = item['wide_initial_target']
        mut_sequence = item['wide_mutated_target']
        
        protospacer_location = ast.literal_eval(item['protospacerlocation_only_initial'])
        pbs_location = ast.literal_eval(item['PBSlocation'])
        rtt_location_wt = ast.literal_eval(item['RT_initial_location'])
        rtt_location_mut = ast.literal_eval(item['RT_mutated_location'])

        protospacer_location_l = protospacer_location[0]
        protospacer_location_r = protospacer_location[1]
        pbs_location_l = pbs_location[0]
        pbs_location_r = pbs_location[1]
        rtt_location_wt_l = rtt_location_wt[0]
        rtt_location_wt_r = rtt_location_wt[1]
        rtt_location_mut_l = rtt_location_mut[0]
        rtt_location_mut_r = rtt_location_mut[1]

        mut_type = mutation_types.index(item['Mutation_Type'])
        if mut_type == 3: # deletion
            mut_type = 2
        elif mut_type == 2: # insertion
            mut_type = 1
        else: # replacement
            mut_type = 0

        rha_length = len(item['RTToverhang'])
        edit_length = int(item['Correction_Length'])

        if mut_type != 2: # not deletion
            lha_length = rtt_location_mut_r - rtt_location_mut_l - rha_length - edit_length
            lha_location_l = rtt_location_wt_l
            lha_location_r = rtt_location_wt_l + lha_length
            rha_location_wt_l = rtt_location_wt_r - rha_length
            rha_location_wt_r = rtt_location_wt_r
            rha_location_mut_l = rtt_location_mut_r - rha_length
            rha_location_mut_r = rtt_location_mut_r
        else:
            lha_length = rtt_location_mut_r - rtt_location_mut_l - rha_length
            lha_location_l = rtt_location_wt_l
            lha_location_r = rtt_location_wt_l + lha_length
            rha_location_wt_l = rtt_location_wt_r - rha_length
            rha_location_wt_r = rtt_location_wt_r
            rha_location_mut_l = rtt_location_mut_r - rha_length
            rha_location_mut_r = rtt_location_mut_r
        spcas9_score = float(item['deepcas9'])
        
        wt_sequence, mut_sequence = align_wt_mut_sequences(wt_sequence, mut_sequence, lha_location_r, edit_length=edit_length, edit_type=mut_type)
        
        rtt_location_l = rtt_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rtt_location_r = rtt_location_wt_r
        else:
            rtt_location_r = rtt_location_mut_r
            
        rha_location_l = rha_location_wt_l
        if mut_type == 2: # deletion, mut sequence is padded with N
            rha_location_r = rha_location_wt_r
        else:
            rha_location_r = rha_location_mut_r

        item_nan = item.isna()

        for cell_line in cell_lines:
            if not item_nan[cell_line]:
                output.append([cell_lines[cell_line], group_id, mut_type, edit_length, wt_sequence, mut_sequence, protospacer_location_l, protospacer_location_r, pbs_location_l, pbs_location_r, rtt_location_l, rtt_location_r, lha_location_l, lha_location_r, rha_location_l, rha_location_r, spcas9_score, item[cell_line]])


    # save the extracted information
    output_df = pd.DataFrame(output, columns=result_columns)
    # each cell line needs to be saved separately
    for cell_line in cell_lines.values():
        target = f"std-pd-{cell_line}-pe2.csv"
        cell_line_data = output_df[output_df['cell-line'] == cell_line]
        # add fold column
        cell_line_data = k_fold_cross_validation_split(cell_line_data, 5)
        cell_line_data.to_csv(DATA_ROOT / 'std' / target, index=False)
    


def std_to_crispai(cell_line: str, pe_system: str, model: str) -> str:
    """
    Convert a standardized format data into crispai format.
    """    
    std_file_name = f'std-{model.lower()}-{cell_line.lower()}-{pe_system.lower()}.csv'
    std_file_path = DATA_ROOT / 'std' / std_file_name
    if not std_file_path.exists():
        raise FileNotFoundError(f"Standardized data file {std_file_name} does not exist in {DATA_ROOT / 'std'}.")
    df = pd.read_csv(std_file_path)
    
    # total read count, unedited percentage and indel percentage are all NaNs,
    df['total_read_count'] = None
    df['edited_percentage'] = df['editing-efficiency']
    df['unedited_percentage'] = None
    df['indel_percentage'] = None
    
    # initial and mutated sequences
    df['initial_sequence'] = df['wt-sequence']
    df['mutated_sequence'] = df['mut-sequence']
    
    # merge the l and r columns into a single column
    # apply to protospacer, pbs and rtt
    df['protospacer_location'] = df.apply(
        lambda row: f"[{row['protospacer-location-l']}, {row['protospacer-location-r']-1}]",
        axis=1
    )
    df['pbs_location'] = df.apply(
        lambda row: f"[{row['pbs-location-l']}, {row['pbs-location-r']-1}]",
        axis=1
    )
    df['rt_initial_location'] = df.apply(
        lambda row: f"[{row['rtt-location-l']}, {row['rtt-location-r']-1}]",
        axis=1
    )
    # rt_mutated_location is the same as the initial locations
    df['rt_mutated_location'] = df['rt_initial_location']
    
    # save the formatted data into the crispai directory
    crispai_path = DATA_ROOT / 'crispai'
    crispai_path.mkdir(parents=True, exist_ok=True)
    file_name = f'crispai-{model.lower()}-{cell_line.lower()}-{pe_system.lower()}.csv'
    crispai_full_path = crispai_path / file_name
    
    # split to train and test based on the value of 'fold'
    df['fold'] = df['fold'].astype(str)
    train_df = df[df['fold'] != 'Test']
    test_df = df[df['fold'] == 'Test']
    crispai_full_path_train = crispai_full_path.with_name(crispai_full_path.stem + '_train.csv')
    crispai_full_path_test = crispai_full_path.with_name(crispai_full_path.stem + '_test.csv')
    
    train_df = train_df[['total_read_count', 'edited_percentage', 'unedited_percentage', 'indel_percentage',
                        'initial_sequence', 'mutated_sequence', 'protospacer_location', 'pbs_location',
                        'rt_initial_location', 'rt_mutated_location']]
    test_df = test_df[['total_read_count', 'edited_percentage', 'unedited_percentage', 'indel_percentage',
                        'initial_sequence', 'mutated_sequence', 'protospacer_location', 'pbs_location',
                        'rt_initial_location', 'rt_mutated_location']]
    
    train_df.to_csv(crispai_full_path_train, index=False)
    test_df.to_csv(crispai_full_path_test, index=False)     
    
    
def std_to_oped(cell_line: str, pe_system: str, model: str) -> str:
    """
    Convert a standardized format data into OPED format.
    
    OPED only takes the sequences data with three columns in a csv file:
    'Target(47bp)', 'PBS', 'RT'
    """
    file_name = f'oped-{model.lower()}-{cell_line.lower()}-{pe_system.lower()}.csv'
    file_path = DATA_ROOT / library_study / 'oped' / file_name
    if file_path.exists():
        print(f"File {file_name} already exists in {DATA_ROOT / library_study / 'oped'}.")
        return pd.read_csv(file_path)
    
    std_file_name = f'std-{model.lower()}-{cell_line.lower()}-{pe_system.lower()}.csv'
    std_file_path = DATA_ROOT / 'std' / std_file_name
    if not std_file_path.exists():
        raise FileNotFoundError(f"Standardized data file {std_file_name} does not exist in {DATA_ROOT / 'std'}.") 
    df = pd.read_csv(std_file_path, nrows=100, dtype={
        'wt-sequence': str, 'mut-sequence': str,
        'protospacer-location-l': int, 'protospacer-location-r': int,
        'pbs-location-l': int, 'pbs-location-r': int,
        'rtt-location-l': int, 'rtt-location-r': int,
        'lha-location-l': int, 'lha-location-r': int,
        'rha-location-l': int, 'rha-location-r': int
    })
    
    # local the wt-sequence, PBS and RT columns
    wt_sequences = df['wt-sequence']
    pbs_sequences = df.apply(
        lambda row: row['mut-sequence'][row['pbs-location-l']:row['pbs-location-r']],
        axis=1
    )
    rtt_sequences = df.apply(
        lambda row: row['mut-sequence'][row['rtt-location-l']:row['rtt-location-r']],
        axis=1
    )
    # remove padding from the sequences
    wt_sequences = [remove_padding(seq) for seq in wt_sequences]
    pbs_sequences = [remove_padding(seq) for seq in pbs_sequences]
    rtt_sequences = [remove_padding(seq) for seq in rtt_sequences]
    
    # invert the rtt and pbs sequences
    pbs_sequence = [pbs[::-1] for pbs in pbs_sequences]
    rtt_sequence = [rtt[::-1] for rtt in rtt_sequences]

    # target sequence is the first 47 bases of the wt-sequence
    target_sequences = [wt[:47] for wt in wt_sequences]
    
    # create a new DataFrame with the required columns
    oped_df = pd.DataFrame({
        'Target(47bp)': target_sequences,
        'PBS': pbs_sequence,
        'RT': rtt_sequence
    })
    if not os.path.exists(DATA_ROOT / library_study / 'oped'):
        os.makedirs(DATA_ROOT / library_study / 'oped')
    oped_df.to_csv(DATA_ROOT / library_study / 'oped' / file_name, index=False)
    print(f"OPED data saved to {DATA_ROOT / library_study / 'oped' / file_name}.")
    
# ==============================================================================
# Data loading functions 
# load data for each model's evaluator
# ==============================================================================
src_model_to_directory = {
    'dp': 'deepprime',
    'dp-ft': 'deepprime',
    'pd2': 'pridict2',
    'oped': 'oped',
    'pd1': 'pridict1',
    'crispai': 'crispai',
    'std': 'std',
}

def load_data(
        cell_line: str, pe_system: str, src_model: str, target_model: str
    ) -> pd.DataFrame:
    filename = f'{target_model}-{src_model}-{cell_line.lower()}-{pe_system.lower()}.csv'
    filepath = DATA_ROOT / src_model_to_directory[target_model] / filename
    if not filepath.exists():
        # convert from std format to the target model format
        conversion_func_name = f'std_to_{src_model_to_directory[target_model]}'
        conversion_func = globals().get(conversion_func_name)
        if conversion_func:
            conversion_func(cell_line, pe_system, src_model)
        else:
            print(f"Conversion function {conversion_func_name} not found.")
            return None
    # load the data after conversion
    df = pd.read_csv(filepath, dtype=str)
    # TODO: convert the columns to the correct types

    return df

# =============================================================================
# Utility functions 
# other data processing functions for training, testing and evaluation
# =============================================================================

def k_fold_cross_validation_split(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """split data into k+1 folds for cross validation.
    

    Args:
        df (pd.DataFrame): DataFrame to be split into folds. Must contain a 'group-id' column.
                            The group-id groups together targets at the same loci to prevent
                            data leakage across folds.
        k (int, optional): Number of folds. Defaults to 5. An additional fold is created for 
                            testing, making it k+1 folds in total.

    Returns:
        pd.DataFrame: DataFrame with an additional 'fold' column.
    """
    df['fold'] = 0
    for f in range(k+1):
        fold_data = df[df['group-id'] % df == f]
        df.loc[fold_data.index, 'fold'] = f if f < k else 'Test'
    
    return df
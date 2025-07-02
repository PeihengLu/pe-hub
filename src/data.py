# src/data.py
# -*- coding: utf-8 -*-
"""Data conversion script for the project.
This script contains functions to convert data from one format to another.
"""
# extract the deep prime original data
import os
import pandas as pd
import numpy as np

from src.constants import DATA_ROOT

def read_from_deepprime_main(original_data_path: str) -> None:
    """
    Read the original deep prime hek293t data to a more usable format.
    Args:
        original_data_path (str): Path to the original deep prime data.
        output_path (str): Path to save the converted data.
    """
    # Load the original data, skipping the first 3 rows and use the forth row as header
    original_data = pd.read_excel(original_data_path, sheet_name='1', skiprows=3, header=0)
    
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

def read_from_deepprime_all(original_data_path: str) -> None:
    """
    Read the original deep prime data to a more usable format.
    Args:
        original_data_path (str): Path to the original deep prime data.
        output_path (str): Path to save the converted data.
    """
    # Load the original data, skipping the first 3 rows and use the forth row as header
    original_data = pd.read_excel(original_data_path, sheet_name='1', skiprows=3, header=0)
    
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

def split_pridict2_orginal_data(original_data_path: str) -> None:
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
    
    

def std_to_crispai(filename: str) -> str:
    """
    Convert a standardized format data into crispai format.
    """
    full_path = DATA_ROOT / 'std'
    full_path = full_path / filename
    df = pd.read_csv(full_path)
    
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
    crispai_full_path = crispai_path / filename
    
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
    
    return str(crispai_full_path)
    
    
    
# TODO: implement load data functions for each model's evaluator
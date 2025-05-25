# src/data.py
# -*- coding: utf-8 -*-
"""Data conversion script for the project.
This script contains functions to convert data from one format to another.
"""
# extract the deep prime original data
import os
import pandas as pd
import numpy as np

def read_from_deepprime_main(original_data_path: str) -> None:
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
    
    
from pathlib import Path
from typing import Optional
import logging
import argparse

import pandas as pd
from tqdm import tqdm
import numpy as np

from pe_common.constants import DATA_ROOT
from pe_common.sequence_utils import align_wt_mut_sequences

def _read_from_deepprime_org(excel_path: Path, sheet_name: str = '1') -> pd.DataFrame:
    """
    Read from the DeepPrime original excel file
    
    Args:
        excel_path: Path to the original deep prime Excel file
        sheet_name: Sheet name to read from
        
    Returns:
        DataFrame with cleaned column names
    """
    # Load the original data, skipping the first 3 rows and use the forth row as header
    original_data = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=3, header=0)
    
    # Rename columns to remove spaces and special characters(newlines, tabs, etc.)
    original_data.columns = (original_data.columns
                            .str.replace(' ', '_')
                            .str.replace('\n', '')
                            .str.replace('\t', ''))
    
    # change the rest of the columns to lower case
    original_data.columns = original_data.columns.str.lower()
    
    # Rename some more opaque columns
    original_data.rename(columns={
        "wide_target_sequence(target_74bps_=_4bp_neighboring_sequence_+_20_bp_protospacer_+_3_bp_ngg_+_47_bp_neighboring_sequence)": "wt-sequence",
        "edited_target_sequence(target_74bps_=_rt-pbs_corresponding_region_and_masked_by_'x')": "mut-sequence",
    }, inplace=True)
    
    return original_data

def export_deepprime_datasheets(original_excel_path: Optional[Path] = None) -> None:
    """
    Export all sheets from the original DeepPrime Excel file to CSV format
    
    Args:
        original_excel_path: Path to the original DeepPrime Excel file.
                            If None, uses default path in raw/deepprime-org/
    """
        
    if original_excel_path is None:
        original_excel_path = DATA_ROOT / 'raw' / 'deepprime-org' / 'deepprime-org.xlsx'
    
    # Read the first sheet and use it as catalog
    original_data_catalog = pd.read_excel(
        original_excel_path, sheet_name='Summary', header=0,
    )
    # Rename the Index Column to 'SheetName'
    original_data_catalog.rename(columns={'Index': 'Sheet name'}, inplace=True)
    
    logger.info(f"Processing {len(original_data_catalog)} sheets from DeepPrime data")

    model_renames = {
        'DeepPrime': 'DeepPrime-Clinvar',
        '-': 'DeepPrime-Off-subpool',
    }
    
    for index, row in original_data_catalog.iterrows():
        # Get the sheet name, cell line, pe system and model 
        sheet_name = row['Sheet name']
        cell_line = row['Cell line']
        pe_system = row['PE system']
        dataset = row['Model']
        if dataset in model_renames:
            dataset = model_renames[dataset]
        
        logger.debug(f"Processing sheet: {sheet_name} ({dataset}, {cell_line}, {pe_system})")
    
        # Load the original data with processing
        original_data = _read_from_deepprime_org(
            original_excel_path, sheet_name=sheet_name
        )
         
        # save the data to a csv file
        output_path = (DATA_ROOT / 'exported' / 'deepprime' / 
                        f"{dataset.lower().replace('-', '_')}-{cell_line.lower().replace('-', '_')}-{pe_system.lower().replace('-', '_')}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

def standardize_pe_data(study, dataset, cell_line, pe_system, data: pd.DataFrame) -> None:
    """
    Standardize the PE data format
    
    Args:
        df: DataFrame in original format
        
    Returns:
        DataFrame in standardized format
    """
    # Placeholder for standardization logic
    if study == 'deepprime':
        _standardize_deepprime(data, cell_line, pe_system, dataset)
        # elif source == 'pridict2':
        #     return _convert_pridict2(cell_line, pe_system)
        # elif source == 'pridict1':
        #     return _convert_pridict(cell_line, pe_system)
    else:
        raise ValueError(f"Unknown study: {study}")

def _standardize_deepprime(
        data: pd.DataFrame, cell_line: str, pe_system: str, dataset: str) -> None:
    '''
    Convert the data from DeepPrime format to the standard format
    '''
    # format the names
    dataset = dataset.lower().replace('-', '_')
    cell_line = cell_line.lower()
    pe_system = pe_system.lower()
    cell_line = cell_line.replace('-', '_')
    pe_system = pe_system.replace('-', '_')
    # output file name
    output_name = f"{dataset}-{cell_line}-{pe_system}.csv"
    
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
    logger.info(f"Standardizing DeepPrime data for {cell_line} - {pe_system} with {len(data)} entries")
    for ind, item in tqdm(data.iterrows(), total=len(data)):
        wt_sequence, mut_sequence = item['wt-sequence'], item['mut-sequence']
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
        else:
            continue
        protospacer_location_l = 5
        protospacer_location_r = 24

        protospacer = wt_sequence[protospacer_location_l:protospacer_location_r]
        
        # grouping by target location, preventing data leakage
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
        original_fold = item['fold'] if 'fold' in item else np.nan
        
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
            spcas9_score, editing_efficiency, original_fold])

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
    output_df['original-fold'] = output_df['original-fold'].astype(str)
    # export the data to a csv file
    output_path = DATA_ROOT / 'standardized' / 'deepprime' / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False) 
    

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # debug mode toggle
    argparser = argparse.ArgumentParser(description="Standardize data from various PE studies")
    argparser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = argparser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

        # exit after setting debug
        exit(0)

    logger.info("Exporting datasheets")
    # export data into data sheets that contain data for one experiement each(model, cell line, pe system)
    deepprime_path = DATA_ROOT / 'raw' / 'deepprime' / 'deepprime-org.xlsx'
    export_deepprime_datasheets(original_excel_path=deepprime_path)
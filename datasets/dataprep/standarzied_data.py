from pathlib import Path
from typing import Optional
import logging
import argparse

import pandas as pd

from pe_common.constants import DATA_ROOT

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

def export_deepprime_all(original_excel_path: Optional[Path] = None) -> None:
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
            _standardize_deepprime(dataset, cell_line, pe_system, data)
        # elif source == 'pridict2':
        #     return _convert_pridict2(cell_line, pe_system)
        # elif source == 'pridict1':
        #     return _convert_pridict(cell_line, pe_system)
    else:
        raise ValueError(f"Unknown study: {study}")

def _standardize_deepprime(
    dataset: str,
    cell_line: str, 
    pe_system: str,
    data: pd.DataFrame
) -> None:
    """
    Standardize DeepPrime data format
    
    Args:
        cell_line: Cell line name
        pe_system: PE system
        data: DataFrame in DeepPrime format
    """
    # Placeholder for actual standardization logic
    standardized = data  # Replace with actual conversion logic

    output_file = (DATA_ROOT / 'standardized' / 'deepprime' / 
                  f'{dataset}-{cell_line.lower()}-{pe_system.lower()}.csv')
    

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
    export_deepprime_all(original_excel_path=deepprime_path)
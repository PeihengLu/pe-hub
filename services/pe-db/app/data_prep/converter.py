"""Data conversion to standardized format for PE Database

This module contains classes and functions to convert various prime editing
dataset formats (DeepPrime, PRIDICT, PRIDICT2, etc.) to a standardized format.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging

from pe_common import DATA_ROOT
from pe_common.sequence_utils import align_wt_mut_sequences, remove_padding

logger = logging.getLogger(__name__)


class DataConverter:
    """Convert various data formats to standardized format"""
    
    # Mapping of model names to data source identifiers
    MODEL_TO_SOURCE = {
        'DeepPrime': 'dp',
        'DeepPrime-FT': 'dp_ft',
    }
    
    def __init__(self, datasets_dir: Optional[Path] = None):
        """
        Initialize DataConverter
        
        Args:
            datasets_dir: Path to datasets directory. Defaults to DATA_ROOT from pe_common
        """
        self.datasets_dir = datasets_dir or DATA_ROOT
        self.raw_dir = self.datasets_dir / 'raw'
        self.std_dir = self.datasets_dir / 'standardized'
        self.std_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataConverter initialized with datasets_dir: {self.datasets_dir}")
    

    
    def convert_to_standardized(
        self, 
        source: str, 
        cell_line: str, 
        pe_system: str,
        model_variant: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert data from a specific source to standardized format
        
        Args:
            source: Data source ('deepprime', 'pridict', 'pridict2', etc.)
            cell_line: Cell line name (e.g., 'HEK293T')
            pe_system: PE system (e.g., 'PE2')
            model_variant: Optional model variant (e.g., 'dp_ft' for DeepPrime-FT)
            
        Returns:
            Standardized DataFrame
        """
        if source == 'deepprime':
            return self._convert_deepprime(cell_line, pe_system, model_variant)
        elif source == 'pridict2':
            return self._convert_pridict2(cell_line, pe_system)
        elif source == 'pridict':
            return self._convert_pridict(cell_line, pe_system)
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _convert_deepprime(
        self, 
        cell_line: str, 
        pe_system: str,
        model_variant: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert DeepPrime format to standardized format
        
        Args:
            cell_line: Cell line name
            pe_system: PE system
            model_variant: Model variant ('dp' or 'dp_ft')
            
        Returns:
            Standardized DataFrame
        """
        # Construct input file path
        variant_str = f"-{model_variant}" if model_variant else ""
        raw_file = (self.datasets_dir / 'deepprime' / 
                   f'deepprime{variant_str}-{cell_line.lower()}-{pe_system.lower()}.csv')
        
        if not raw_file.exists():
            raise FileNotFoundError(f"DeepPrime data file not found: {raw_file}")
        
        df = pd.read_csv(raw_file)
        
        # TODO: Implement conversion logic from src/data.py deepprime_org_to_std
        # This will need to be extracted from the existing src/data.py file
        standardized = df  # Placeholder
        
        # Save to standardized directory
        output_file = (self.std_dir / 'deepprime' / 
                      f'std-dp{variant_str}-{cell_line.lower()}-{pe_system.lower()}.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        standardized.to_csv(output_file, index=False)
        
        logger.info(f"Converted DeepPrime data to {output_file}")
        return standardized
    
    def _convert_pridict2(self, cell_line: str, pe_system: str) -> pd.DataFrame:
        """Convert PRIDICT2 format to standardized format"""
        raw_file = self.raw_dir / 'pridict2' / f'{cell_line}_{pe_system}.csv'
        
        if not raw_file.exists():
            raise FileNotFoundError(f"PRIDICT2 data file not found: {raw_file}")
        
        df = pd.read_csv(raw_file)
        
        # TODO: Implement conversion logic
        standardized = df  # Placeholder
        
        output_file = (self.std_dir / 'pridict2' / 
                      f'std-pd2-{cell_line.lower()}-{pe_system.lower()}.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        standardized.to_csv(output_file, index=False)
        
        logger.info(f"Converted PRIDICT2 data to {output_file}")
        return standardized
    
    def _convert_pridict(self, cell_line: str, pe_system: str) -> pd.DataFrame:
        """Convert PRIDICT format to standardized format"""
        raw_file = self.raw_dir / 'pridict1' / f'{cell_line}_{pe_system}.csv'
        
        if not raw_file.exists():
            raise FileNotFoundError(f"PRIDICT data file not found: {raw_file}")
        
        df = pd.read_csv(raw_file)
        
        # TODO: Implement conversion logic
        standardized = df  # Placeholder
        
        output_file = (self.std_dir / 'pridict1' / 
                      f'std-pd1-{cell_line.lower()}-{pe_system.lower()}.csv')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        standardized.to_csv(output_file, index=False)
        
        logger.info(f"Converted PRIDICT data to {output_file}")
        return standardized

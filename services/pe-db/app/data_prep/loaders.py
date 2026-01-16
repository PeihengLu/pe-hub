"""Data loaders for PE Database

This module provides functionality to load data in various formats
for different prime editing efficiency prediction models.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Literal
import logging

from pe_common import DATA_ROOT

logger = logging.getLogger(__name__)


class PEDataLoader:
    """Load prime editing data in various formats"""
    
    def __init__(self, datasets_dir: Optional[Path] = None):
        """
        Initialize PEDataLoader
        
        Args:
            datasets_dir: Path to datasets directory. Defaults to DATA_ROOT from pe_common
        """
        self.datasets_dir = datasets_dir or DATA_ROOT
        self.std_dir = self.datasets_dir / 'standardized'
        
        logger.info(f"DataLoader initialized with datasets_dir: {self.datasets_dir}")
    
    def load_data(
        self,
        cell_line: str,
        pe_system: str,
        source_model: str,
        target_format: Literal["std", "oped", "deepprime", "pridict", "pridict2"] = "std"
    ) -> pd.DataFrame:
        """
        Load data in the requested format
        
        Args:
            cell_line: Cell line name (e.g., 'HEK293T')
            pe_system: PE system (e.g., 'PE2')
            source_model: Source model identifier ('dp', 'dp_ft', 'pd1', 'pd2', etc.)
            target_format: Target format to load data in
            
        Returns:
            DataFrame with data in the requested format
            
        Raises:
            FileNotFoundError: If the requested data file doesn't exist
        """
        # Normalize inputs
        cell_line_lower = cell_line.lower()
        pe_system_lower = pe_system.lower()
        
        if target_format == "std":
            # Load standardized format
            file_path = self._find_standardized_file(cell_line_lower, pe_system_lower, source_model)
        elif target_format == "deepprime":
            # Load DeepPrime format
            file_path = self._find_deepprime_format_file(cell_line_lower, pe_system_lower, source_model)
        elif target_format == "pridict":
            # Load PRIDICT format
            file_path = self._find_pridict_format_file(cell_line_lower, pe_system_lower, source_model, version=1)
        elif target_format == "pridict2":
            # Load PRIDICT2 format
            file_path = self._find_pridict_format_file(cell_line_lower, pe_system_lower, source_model, version=2)
        elif target_format == "oped":
            # Load OPED format
            file_path = self._find_oped_format_file(cell_line_lower, pe_system_lower, source_model)
        else:
            raise ValueError(f"Unknown target format: {target_format}")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Parameters: cell_line={cell_line}, pe_system={pe_system}, "
                f"source_model={source_model}, format={target_format}"
            )
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    
    def _find_standardized_file(self, cell_line: str, pe_system: str, source_model: str) -> Path:
        """Find standardized format file"""
        # Map source model to directory and filename pattern
        if source_model in ['dp', 'dp_ft']:
            source_dir = 'deepprime'
        elif source_model == 'pd1':
            source_dir = 'pridict1'
        elif source_model == 'pd2':
            source_dir = 'pridict2'
        else:
            source_dir = source_model
        
        file_path = (self.std_dir / source_dir / 
                    f'std-{source_model}-{cell_line}-{pe_system}.csv')
        return file_path
    
    def _find_deepprime_format_file(self, cell_line: str, pe_system: str, source_model: str) -> Path:
        """Find DeepPrime format file"""
        # Check if we have a pre-formatted version
        file_path = (self.datasets_dir / 'deepprime' / 
                    f'{source_model}-{cell_line}-{pe_system}.csv')
        return file_path
    
    def _find_pridict_format_file(
        self, 
        cell_line: str, 
        pe_system: str, 
        source_model: str,
        version: int = 1
    ) -> Path:
        """Find PRIDICT format file"""
        dir_name = f'pridict{version}'
        file_path = (self.datasets_dir / dir_name / 
                    f'{source_model}-{cell_line}-{pe_system}.csv')
        return file_path
    
    def _find_oped_format_file(self, cell_line: str, pe_system: str, source_model: str) -> Path:
        """Find OPED format file"""
        file_path = (self.datasets_dir / 'oped' / 
                    f'{source_model}-{cell_line}-{pe_system}.csv')
        return file_path
    
    def list_available_datasets(self) -> pd.DataFrame:
        """
        List all available datasets in the standardized directory
        
        Returns:
            DataFrame with columns: source, cell_line, pe_system, file_path
        """
        datasets = []
        
        # Scan standardized directory
        if self.std_dir.exists():
            for source_dir in self.std_dir.iterdir():
                if not source_dir.is_dir():
                    continue
                
                for csv_file in source_dir.glob('*.csv'):
                    # Parse filename: std-{source}-{cell_line}-{pe_system}.csv
                    parts = csv_file.stem.split('-')
                    if len(parts) >= 4 and parts[0] == 'std':
                        datasets.append({
                            'source': parts[1],
                            'cell_line': parts[2],
                            'pe_system': parts[3],
                            'file_path': str(csv_file)
                        })
        
        return pd.DataFrame(datasets)

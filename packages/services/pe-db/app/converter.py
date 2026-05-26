"""Data conversion from standardized format for PE Database

This module contains classes and functions to convert various prime editing
dataset formats (DeepPrime, PRIDICT, PRIDICT2, etc.) to a standardized format.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

from pe_common.constants import DATA_ROOT
from .utils.convert_data import (
    is_standardized_dataframe,
    standardized_to_pridict_dataframe,
    standardized_to_deepprime_dataframe,
    standardized_to_oped_dataframe,
)
from .utils.standardize_data import export_original_data, standardize_pe_data

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
        self.exported_dir = self.datasets_dir / 'exported'
        self.exported_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataConverter initialized with datasets_dir: {self.datasets_dir}")

    def export_raw(
        self,
        study: Optional[str] = None,
        *,
        force_reexport: bool = False,
    ) -> None:
        """Export raw study files to ``datasets/exported/`` and index Datasheet catalog rows."""
        export_original_data(study=study, force_reexport=force_reexport)

    def initialize_database(
        self,
        *,
        force_export: bool = False,
        force_standardize: bool = False,
    ) -> None:
        """Seed catalog, export raw CSVs, standardize to parquet, and index Datasheets."""
        from .catalog.initialize import initialize_database as _initialize_database

        _initialize_database(
            force_export=force_export,
            force_standardize=force_standardize,
        )

    def convert_to_standardized(
        self,
        *,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
    ) -> pd.DataFrame:
        """Standardize one exported datasheet (edit records stay on disk, not in the catalog DB)."""
        return standardize_pe_data(
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
        )
    
    def convert_from_standardized(
        self,
        source: Union[pd.DataFrame, Path],
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
        target_format: str,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Convert standardized data into a model-specific format.

        Args:
            source: Standardized dataframe or path to a standardized CSV file
            target_format: One of {"std", "deepprime", "pridict", "pridict2", "oped"}
            output_file: Optional CSV path to write converted output
        """
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"Standardized data file not found: {source}")
            df = pd.read_parquet(source) # standardized data is always in parquet format
        else:
            df = source.copy()

        if target_format == "std":
            converted = df.copy()
        else:
            if not is_standardized_dataframe(df):
                raise ValueError("Input dataframe is not in standardized schema.")
            if target_format == "deepprime":
                converted = standardized_to_deepprime_dataframe(df)
            elif target_format in {"pridict", "pridict2"}:
                converted = standardized_to_pridict_dataframe(df)
            elif target_format == "oped":
                converted = standardized_to_oped_dataframe(df)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")

        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.suffix.lower() == ".parquet":
                converted.to_parquet(output_file, index=False)
            else:
                converted.to_csv(output_file, index=False)
            logger.info(f"Converted standardized data to {target_format}: {output_file}")

        return converted
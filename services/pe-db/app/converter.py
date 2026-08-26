"""Data conversion from standardized format for PE Database

This module contains classes and functions to convert various prime editing
dataset formats (DeepPrime, PRIDICT, PRIDICT2, etc.) to a standardized format.
"""
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Union
import logging

from pe_common.constants import DATA_ROOT
from .format_registry import convert_standardized, known_model_formats
from .formatted_cache import (
    load_formatted_cache,
    save_formatted_cache,
)
from .utils.convert_data import is_standardized_dataframe
from .utils.standardize_data import export_original_data, standardize_pe_data

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


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
        self.formatted_dir = self.datasets_dir / 'formatted'
        self.formatted_dir.mkdir(parents=True, exist_ok=True)

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
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """
        Convert standardized data into a model-specific format.

        Args:
            source: Standardized dataframe or path to a standardized CSV file
            target_format: Registered output format (see ``format_registry``)
            output_file: Optional CSV path to write converted output
        """
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"Standardized data file not found: {source}")
            df = pd.read_parquet(source) # standardized data is always in parquet format
        else:
            df = source.copy()

        if target_format != "std" and not is_standardized_dataframe(df):
            raise ValueError("Input dataframe is not in standardized schema.")

        converted = convert_standardized(
            df,
            target_format,
            progress_callback=progress_callback,
        )

        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.suffix.lower() == ".parquet":
                converted.to_parquet(output_file, index=False)
            else:
                converted.to_csv(output_file, index=False)
            logger.info(f"Converted standardized data to {target_format}: {output_file}")

        return converted

    def load_or_convert_formatted(
        self,
        source: pd.DataFrame,
        *,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
        target_format: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Return model-format rows for a full standardized datasheet, using disk cache."""
        if target_format == "std":
            return source.copy()
        if target_format not in known_model_formats():
            raise ValueError(f"Unsupported target format: {target_format}")

        cached = load_formatted_cache(
            target_format,
            study,
            dataset,
            cell_line,
            pe_system,
            datasets_dir=self.datasets_dir,
            expected_rows=len(source),
        )
        if cached is not None:
            # Parquet round-trip drops the index; restore source row labels so
            # callers can subset with ``.loc[filtered.index]`` after edit filters.
            cached = cached.copy()
            cached.index = source.index
            if progress_callback is not None:
                progress_callback(
                    f"Loaded formatted cache for {target_format} "
                    f"({study}/{dataset} · {cell_line} · {pe_system}, {len(cached)} rows)"
                )
            return cached

        if progress_callback is not None:
            progress_callback(
                f"Converting {len(source)} standardized rows to {target_format} "
                f"for {study}/{dataset} · {cell_line} · {pe_system}"
            )

        converted = self.convert_from_standardized(
            source,
            study=study,
            dataset=dataset,
            cell_line=cell_line,
            pe_system=pe_system,
            target_format=target_format,
            progress_callback=progress_callback,
        )
        save_formatted_cache(
            converted,
            target_format,
            study,
            dataset,
            cell_line,
            pe_system,
            datasets_dir=self.datasets_dir,
        )
        return converted

"""Convert standardized datasheets into registered model formats."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from pe_common.constants import DATA_ROOT

from .format_registry import convert_standardized, known_model_formats
from .formatted_cache import load_formatted_cache, save_formatted_cache
from .pipeline.run import export_original_data, standardize_pe_data
from .formats.common import is_standardized_dataframe

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


class DataConverter:
    """Cached standardized → model-format conversion for one datasheet."""

    def __init__(self, datasets_dir: Optional[Path] = None):
        self.datasets_dir = datasets_dir or DATA_ROOT
        self.std_dir = self.datasets_dir / "standardized"
        self.std_dir.mkdir(parents=True, exist_ok=True)
        self.exported_dir = self.datasets_dir / "exported"
        self.exported_dir.mkdir(parents=True, exist_ok=True)
        self.formatted_dir = self.datasets_dir / "formatted"
        self.formatted_dir.mkdir(parents=True, exist_ok=True)

    def export_raw(
        self,
        study: Optional[str] = None,
        *,
        force_reexport: bool = False,
    ) -> None:
        export_original_data(study=study, force_reexport=force_reexport)

    def convert_to_standardized(
        self,
        *,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
    ) -> pd.DataFrame:
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
        del study, dataset, cell_line, pe_system
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"Standardized data file not found: {source}")
            df = pd.read_parquet(source)
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
            logger.info("Converted standardized data to %s: %s", target_format, output_file)

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

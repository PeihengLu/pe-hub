"""Load standardized PE-DB parquet datasheets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from pe_common.constants import DATA_ROOT

from .pipeline.names import _normalize_name

logger = logging.getLogger(__name__)


class PEDataLoader:
    """Load PE data using hierarchy: study/dataset/cell_line-pe_system."""

    def __init__(self, datasets_dir: Optional[Path] = None):
        self.datasets_dir = datasets_dir or DATA_ROOT
        self.std_dir = self.datasets_dir / "standardized"

    def load_data(
        self,
        *,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
    ) -> pd.DataFrame:
        study = _normalize_name(study)
        dataset = _normalize_name(dataset)
        cell_line = _normalize_name(cell_line)
        pe_system = _normalize_name(pe_system)

        file_path = self._find_standardized_file(
            study=study, dataset=dataset, cell_line=cell_line, pe_system=pe_system
        )
        if not file_path.exists():
            raise FileNotFoundError(
                f"Standardized data file not found: {file_path}\n"
                f"Parameters: study={study}, dataset={dataset}, cell_line={cell_line}, "
                f"pe_system={pe_system}"
            )
        logger.info("Loading data from %s", file_path)
        return self._read_dataframe(file_path)

    @staticmethod
    def _read_dataframe(file_path: Path) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    def _find_standardized_file(
        self, *, study: str, dataset: str, cell_line: str, pe_system: str
    ) -> Path:
        study = _normalize_name(study)
        dataset = _normalize_name(dataset)
        cell_line = _normalize_name(cell_line)
        pe_system = _normalize_name(pe_system)
        stem = f"{cell_line}-{pe_system}"
        return self.std_dir / study / dataset / f"{stem}.parquet"

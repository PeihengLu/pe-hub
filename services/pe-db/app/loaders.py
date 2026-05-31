"""Data loaders for PE Database."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from pe_common.constants import DATA_ROOT
logger = logging.getLogger(__name__)


def _normalize_name(value: str, *, replace_hyphen: bool = True) -> str:
    normalized = str(value).strip().lower()
    if replace_hyphen:
        return normalized.replace("-", "_")
    return normalized


class PEDataLoader:
    """Load PE data using hierarchy: study/dataset/cell_line-pe_system."""

    def __init__(self, datasets_dir: Optional[Path] = None):
        self.datasets_dir = datasets_dir or DATA_ROOT
        self.raw_dir = self.datasets_dir / "raw"
        self.std_dir = self.datasets_dir / "standardized"
        self.exported_dir = self.datasets_dir / "exported"

        logger.info("DataLoader initialized with datasets_dir=%s", self.datasets_dir)

    def load_data(
        self,
        *,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
        target_format: Literal["std", "oped", "deepprime", "pridict", "pridict2"] = "std",
    ) -> pd.DataFrame:
        """
        Load data from standardized or model-specific format.

        Naming hierarchy:
        - standardized: standardized/{study}/{dataset}/{cell_line}-{pe_system}.parquet
        - model format: {format}/{study}/{dataset}/{cell_line}-{pe_system}.csv
        """
        study = _normalize_name(study)
        dataset = _normalize_name(dataset)
        cell_line = _normalize_name(cell_line)
        pe_system = _normalize_name(pe_system)

        if target_format == "std":
            file_path = self._find_standardized_file(
                study=study, dataset=dataset, cell_line=cell_line, pe_system=pe_system
            )
        else:
            file_path = self._find_model_format_file(
                target_format=target_format,
                study=study,
                dataset=dataset,
                cell_line=cell_line,
                pe_system=pe_system,
            )

        if not file_path.exists():
            if target_format != "std":
                std_file = self._find_standardized_file(
                    study=study, dataset=dataset, cell_line=cell_line, pe_system=pe_system
                )
                if std_file.exists():
                    from .converter import DataConverter

                    converter = DataConverter(self.datasets_dir)
                    logger.info(
                        "Model-format file missing; generating %s from standardized file %s",
                        target_format,
                        std_file,
                    )
                    return converter.convert_from_standardized(
                        std_file,
                        study=study,
                        dataset=dataset,
                        cell_line=cell_line,
                        pe_system=pe_system,
                        target_format=target_format,
                        output_file=file_path,
                    )

            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Parameters: study={study}, dataset={dataset}, cell_line={cell_line}, "
                f"pe_system={pe_system}, format={target_format}"
            )

        logger.info("Loading data from %s", file_path)
        return self._read_dataframe(file_path)

    def _find_standardized_file(
        self, *, study: str, dataset: str, cell_line: str, pe_system: str
    ) -> Path:
        """Find standardized file (prefer parquet, fallback csv)."""
        study = _normalize_name(study)
        dataset = _normalize_name(dataset)
        cell_line = _normalize_name(cell_line)
        pe_system = _normalize_name(pe_system)
        stem = f"{cell_line}-{pe_system}"
        parquet_file = self.std_dir / study / dataset / f"{stem}.parquet"
        if parquet_file.exists():
            return parquet_file
        return self.std_dir / study / dataset / f"{stem}.csv"

    def _find_model_format_file(
        self,
        *,
        target_format: str,
        study: str,
        dataset: str,
        cell_line: str,
        pe_system: str,
    ) -> Path:
        stem = f"{cell_line}-{pe_system}"
        return self.datasets_dir / target_format / study / dataset / f"{stem}.csv"

    @staticmethod
    def _read_dataframe(file_path: Path) -> pd.DataFrame:
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def list_available_datasets(self) -> pd.DataFrame:
        """
        List all standardized datasets.

        Returns columns: study, dataset, cell_line, pe_system, format, file_path
        """
        rows: list[dict[str, str]] = []
        if not self.std_dir.exists():
            return pd.DataFrame(rows)

        for data_file in self.std_dir.rglob("*"):
            if data_file.suffix.lower() not in {".parquet", ".csv"} or not data_file.is_file():
                continue

            rel_parts = data_file.relative_to(self.std_dir).parts
            if len(rel_parts) != 3:
                continue
            study, dataset, _filename = rel_parts
            if "-" not in data_file.stem:
                continue
            cell_line, pe_system = data_file.stem.rsplit("-", 1)
            rows.append(
                {
                    "study": study,
                    "dataset": dataset,
                    "cell_line": cell_line,
                    "pe_system": pe_system,
                    "format": data_file.suffix.lstrip("."),
                    "file_path": str(data_file),
                }
            )
        return pd.DataFrame(rows)


DataLoader = PEDataLoader
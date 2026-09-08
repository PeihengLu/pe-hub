"""Catalog indexing must accept hyphen or underscore export directory names."""
from __future__ import annotations

from pathlib import Path

from pe_db.catalog.datasheets import _normalize_dataset_name, index_exported_datasheets
from pe_db.library import list_datasheets


def test_normalize_dataset_name_canonicalizes_underscores() -> None:
    assert _normalize_dataset_name("trip_analysis") == "trip-analysis"
    assert _normalize_dataset_name("trip-analysis") == "trip-analysis"


def test_underscore_export_dir_indexes_hyphen_catalog_name(
    seeded_catalog: Path,
) -> None:
    csv_dir = seeded_catalog / "exported" / "pridict2" / "trip_analysis"
    csv_dir.mkdir(parents=True)
    (csv_dir / "k562-pe2.csv").write_text("barcode,PE_editing_efficiency\nx,1.0\n")

    index_exported_datasheets(data_root=seeded_catalog)
    rows = list_datasheets(study="pridict2", dataset="trip-analysis")
    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "trip-analysis"
    assert rows[0]["cell_line"] == "k562"
    assert rows[0]["pe_system"] == "pe2"

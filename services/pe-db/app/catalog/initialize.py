"""Full PE Database catalog initialization pipeline."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def initialize_database(
    *,
    force_export: bool = False,
    force_standardize: bool = False,
) -> None:
    """
    Run the full catalog + data preparation pipeline.

    1. **Seed** — create tables and load hand-maintained Study, Dataset, Scaffold rows
    2. **Export** — write ``datasets/exported/`` CSVs from raw sources; index Datasheet rows
    3. **Standardize** — write ``datasets/standardized/`` parquet files from exported CSVs
    """
    from ..utils.standardize_data import export_original_data, standardize_exported_data
    from .seed import init_catalog

    logger.info("PE Database initialization: seeding catalog")
    init_catalog()

    logger.info("PE Database initialization: exporting raw data")
    export_original_data(force_reexport=force_export)

    logger.info("PE Database initialization: standardizing exported data")
    standardize_exported_data(force=force_standardize)

    logger.info("PE Database initialization complete")

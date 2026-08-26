"""Spawn-safe entry points for PRIDICT2 MFE conversion workers.

``services/pe-db/app`` is imported as ``pe_db_service_app`` when pe-ensemble is
in the same process. Spawn children cannot import that runtime alias, so the
process-pool initializer and submitted callable must live in this installable
``pe_db`` package.
"""
from __future__ import annotations

from pe_db._bootstrap import ensure_service_app_importable, import_service_app


def init_mfe_worker() -> None:
    """Register ``pe_db_service_app`` and preload the converter in a spawn worker."""
    ensure_service_app_importable()
    import_service_app("utils.convert_data")


def pridict2_mfe_chunk_worker(
    chunk: list[tuple[str, str, dict[str, int]]],
) -> list[dict[str, float]]:
    convert_data = import_service_app("utils.convert_data")
    return convert_data._pridict2_mfe_chunk_worker(chunk)

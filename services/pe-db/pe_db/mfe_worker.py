"""Spawn-safe entry points for PRIDICT2 MFE conversion workers.

The initializer and submitted callable live in this installable ``pe_db``
package so spawn children can unpickle them.
"""
from __future__ import annotations


def init_mfe_worker() -> None:
    """Preload ViennaRNA and the PRIDICT converter in a spawn worker."""
    from pe_db.formats import pridict  # noqa: F401


def pridict2_mfe_chunk_worker(
    chunk: list[tuple[str, str, dict[str, int]]],
) -> list[dict[str, float]]:
    from pe_db.formats.pridict import _pridict2_mfe_chunk_worker

    return _pridict2_mfe_chunk_worker(chunk)

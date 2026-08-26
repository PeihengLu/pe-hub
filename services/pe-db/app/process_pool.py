"""Shared process pool for CPU-heavy conversion work."""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

_pool: Optional[ProcessPoolExecutor] = None


def mfe_worker_count() -> int:
    configured = os.getenv("PRIDICT2_MFE_WORKERS", "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, os.cpu_count() or 1)


def get_mfe_process_pool() -> ProcessPoolExecutor:
    """Return a long-lived spawn-based pool for PRIDICT2 MFE conversion."""
    global _pool
    if _pool is None:
        # Initializer lives in the installable ``pe_db`` package so spawn
        # workers can unpickle it (they cannot import ``pe_db_service_app``).
        from pe_db.mfe_worker import init_mfe_worker

        _pool = ProcessPoolExecutor(
            max_workers=mfe_worker_count(),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_mfe_worker,
        )
    return _pool


def shutdown_mfe_process_pool() -> None:
    global _pool
    if _pool is None:
        return
    _pool.shutdown(wait=True, cancel_futures=False)
    _pool = None

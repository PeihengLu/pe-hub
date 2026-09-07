"""Spawn workers must unpickle callables from the installable ``pe_db`` package."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from pe_db.formats.pridict import _pridict2_mfe_chunk_worker
from pe_db.mfe_worker import init_mfe_worker, pridict2_mfe_chunk_worker


def test_mfe_chunk_worker_lives_in_installable_package():
    assert pridict2_mfe_chunk_worker.__module__ == "pe_db.mfe_worker"
    assert _pridict2_mfe_chunk_worker.__module__ == "pe_db.formats.pridict"


def test_spawn_pool_runs_pickle_safe_mfe_worker():
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=init_mfe_worker,
    ) as pool:
        assert pool.submit(pridict2_mfe_chunk_worker, []).result() == []


def test_spawn_pool_can_unpickle_format_mfe_worker():
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=init_mfe_worker,
    ) as pool:
        assert pool.submit(_pridict2_mfe_chunk_worker, []).result() == []

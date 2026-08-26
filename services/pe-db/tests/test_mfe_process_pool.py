"""Spawn workers must not depend on the runtime ``pe_db_service_app`` alias."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from pe_db._bootstrap import import_service_app
from pe_db.mfe_worker import init_mfe_worker, pridict2_mfe_chunk_worker


def test_mfe_chunk_worker_lives_in_installable_package():
    assert pridict2_mfe_chunk_worker.__module__ == "pe_db.mfe_worker"


def test_spawn_pool_runs_pickle_safe_mfe_worker():
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=init_mfe_worker,
    ) as pool:
        assert pool.submit(pridict2_mfe_chunk_worker, []).result() == []


def test_spawn_pool_can_unpickle_aliased_convert_data_worker():
    convert_data = import_service_app("utils.convert_data")
    worker = convert_data._pridict2_mfe_chunk_worker
    assert worker.__module__.startswith("pe_db_service_app")

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=init_mfe_worker,
    ) as pool:
        assert pool.submit(worker, []).result() == []

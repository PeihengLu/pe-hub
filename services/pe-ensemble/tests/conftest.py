"""Pytest path setup for pe-ensemble tests.

Puts ``services/pe-ensemble`` on ``sys.path`` so ``pe_ensemble`` imports resolve
without an editable install. Paths are applied only while ensemble tests are
being collected, so pe-db and pe-ensemble suites can coexist in one pytest
invocation.

For isolated runs, ``scripts/run-tests.sh`` is the recommended entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_PE_COMMON = Path(__file__).resolve().parents[3] / "packages" / "pe-common"


def _ensemble_path_in_sys_path() -> None:
    for entry in (_SERVICE_ROOT, _PE_COMMON):
        path = str(entry)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _collector_path(collector) -> str:
    path = getattr(collector, "path", None) or getattr(collector, "fspath", None)
    return str(path).replace("\\", "/")


def _is_under_pe_ensemble_tests(collector) -> bool:
    tests_root = _SERVICE_ROOT / "tests"
    node = collector
    while node is not None:
        path = Path(_collector_path(node))
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved == tests_root or tests_root in resolved.parents:
            return True
        node = getattr(node, "parent", None)
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector) -> None:
    if not _is_under_pe_ensemble_tests(collector):
        return
    _ensemble_path_in_sys_path()


@pytest.hookimpl(tryfirst=True)
def pytest_make_collect_report(collector):
    if not _is_under_pe_ensemble_tests(collector):
        return None
    path = _collector_path(collector)
    if not path.endswith(".py"):
        return None
    _ensemble_path_in_sys_path()
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    if not _is_under_pe_ensemble_tests(item):
        return
    _ensemble_path_in_sys_path()


@pytest.fixture(autouse=True)
def _restore_pe_db_access_mode():
    """Keep the in-process/HTTP PE-DB mode from leaking between tests."""
    config_mod = sys.modules.get("pe_ensemble.training.config")
    saved = getattr(config_mod, "_use_pe_db_library", None) if config_mod is not None else None
    yield
    config_mod = sys.modules.get("pe_ensemble.training.config")
    if config_mod is not None and saved is not None:
        config_mod._use_pe_db_library = saved

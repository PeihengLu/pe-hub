"""Pytest path setup for pe-db tests.

Ensures ``app`` resolves to ``services/pe-db/app``. Paths are applied only while
pe-db tests are being collected (not at conftest import time), so pe-db and
pe-ensemble suites can coexist in one pytest invocation.

For isolated runs, ``run-smoke-tests.sh`` is still the recommended entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_PE_COMMON = Path(__file__).resolve().parents[3] / "packages" / "pe-common"
_PE_DB_APP = _SERVICE_ROOT / "app"


def _pe_db_path_in_sys_path() -> None:
    for entry in (_SERVICE_ROOT, _PE_COMMON):
        path = str(entry)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _purge_conflicting_app_modules() -> None:
    app = sys.modules.get("app")
    if app is None:
        return
    app_file = getattr(app, "__file__", "") or ""
    if str(_PE_DB_APP) in app_file:
        return
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]


def _collector_path(collector) -> str:
    path = getattr(collector, "path", None) or getattr(collector, "fspath", None)
    return str(path).replace("\\", "/")


def _is_under_pe_db_tests(collector) -> bool:
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
    if not _is_under_pe_db_tests(collector):
        return
    _pe_db_path_in_sys_path()
    _purge_conflicting_app_modules()


@pytest.hookimpl(tryfirst=True)
def pytest_make_collect_report(collector):
    if not _is_under_pe_db_tests(collector):
        return None
    path = _collector_path(collector)
    if not path.endswith(".py"):
        return None
    _pe_db_path_in_sys_path()
    _purge_conflicting_app_modules()
    return None

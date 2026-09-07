"""Pytest path setup for pe-db tests.

Ensures ``app`` resolves to ``services/pe-db/app``. Paths are applied only while
pe-db tests are being collected (not at conftest import time), so pe-db and
pe-ensemble suites can coexist in one pytest invocation.

For isolated runs, ``scripts/run-tests.sh`` is the recommended entry point.
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


def reset_pe_db_engine() -> None:
    """Drop cached Settings and SQLAlchemy engines for both ``app`` import paths.

    Tests import ``app.*`` directly; ``pedb`` goes through ``pe_db_service_app.*``.
    Both copies cache Settings and a process-global engine.
    """
    for config_name in ("app.config", "pe_db_service_app.config"):
        config_mod = sys.modules.get(config_name)
        if config_mod is not None and hasattr(config_mod, "get_settings"):
            config_mod.get_settings.cache_clear()
    try:
        from app.config import get_settings as _app_get_settings

        _app_get_settings.cache_clear()
    except ImportError:
        pass
    for session_name in ("app.db.session", "pe_db_service_app.db.session"):
        session_mod = sys.modules.get(session_name)
        if session_mod is None:
            continue
        engine = getattr(session_mod, "_engine", None)
        if engine is not None:
            engine.dispose()
        session_mod._engine = None
        session_mod._SessionLocal = None


@pytest.fixture(autouse=True)
def _reset_pe_db_engine_between_tests() -> None:
    """Prevent DATABASE_URL / DATA_ROOT leaks across pe-db tests."""
    reset_pe_db_engine()
    yield
    reset_pe_db_engine()


@pytest.fixture
def seeded_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seed Study/Dataset/Scaffold rows into an isolated sqlite catalog."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PLUGINS_ROOT", str(tmp_path / "plugins"))
    (tmp_path / "plugins").mkdir()
    reset_pe_db_engine()
    from app.library import run_seed

    run_seed()
    return tmp_path

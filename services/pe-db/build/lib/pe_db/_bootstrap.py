"""Ensure pe-db service code is importable without colliding with pe-ensemble ``app``."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
# Private package alias so ``from pe_db.library import ...`` works inside pe-ensemble,
# which also owns a top-level ``app`` package in the same process.
_SERVICE_APP_ALIAS = "pe_db_service_app"


def ensure_service_root_on_path() -> Path:
    root = str(_SERVICE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _SERVICE_ROOT


def _load_service_app_package() -> ModuleType:
    """Load ``services/pe-db/app`` as ``pe_db_service_app`` (collision-safe)."""
    existing = sys.modules.get(_SERVICE_APP_ALIAS)
    if existing is not None:
        return existing

    import importlib.util

    app_dir = _SERVICE_ROOT / "app"
    init_path = app_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _SERVICE_APP_ALIAS,
        init_path,
        submodule_search_locations=[str(app_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pe-db app package from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SERVICE_APP_ALIAS] = module
    spec.loader.exec_module(module)
    return module


def import_service_app(module: str = "library") -> ModuleType:
    """Import a submodule of pe-db's ``app`` package under a private alias.

    Prefer this over ``from app.library import ...`` whenever pe-ensemble (or any
    other package named ``app``) may already be loaded in ``sys.modules``.
    """
    ensure_service_root_on_path()
    _load_service_app_package()
    return importlib.import_module(f"{_SERVICE_APP_ALIAS}.{module}")

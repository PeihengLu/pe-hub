"""Ensure ``services/pe-db`` is on ``sys.path`` so ``app.*`` imports resolve."""
from __future__ import annotations

import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def ensure_service_root_on_path() -> Path:
    root = str(_SERVICE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _SERVICE_ROOT

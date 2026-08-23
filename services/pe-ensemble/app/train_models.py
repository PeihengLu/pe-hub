#!/usr/bin/env python3
"""Backward-compatible shim for ``pe-ensemble train``."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))


def main(argv: Optional[List[str]] = None) -> int:
    from pe_ensemble.cli import main as cli_main

    if argv is None:
        argv = sys.argv[1:]
    return cli_main(["train", *argv])


if __name__ == "__main__":
    raise SystemExit(main())

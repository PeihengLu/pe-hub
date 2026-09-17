"""Built-in study pipelines (export + standardize)."""
from __future__ import annotations

_LOADED = False


def load_studies() -> None:
    global _LOADED
    if _LOADED:
        return
    from . import (  # noqa: F401
        anzalone,
        deeppe,
        deepprime,
        minsepie,
        optiprime,
        pridict1,
        pridict2,
    )

    _LOADED = True

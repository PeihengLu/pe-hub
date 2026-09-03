"""Guard PRIDICT2 weight bundles against config/statedict mismatches."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ROOT = REPO_ROOT / "services" / "pe-ensemble" / "weights" / "pridict2"


def _registered_pridict2_ids() -> list[str]:
    root = WEIGHTS_ROOT
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and (entry / "manifest.json").is_file()
    )


@pytest.mark.parametrize("weight_id", _registered_pridict2_ids())
def test_pridict2_weight_bundle_matches_config(weight_id: str):
    PRIDICT2ModelWrapper._validate_run_dir(str(WEIGHTS_ROOT / weight_id))

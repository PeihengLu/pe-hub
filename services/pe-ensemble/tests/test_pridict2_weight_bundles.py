"""Guard PRIDICT2 weight bundles against config/statedict mismatches."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ROOT = REPO_ROOT / "services" / "pe-ensemble" / "weights" / "pridict2"

KNOWN_BROKEN_IDS = {
    f"pridict1_1__exp_2023-12-22_14-22-03__run_{fold}"
    for fold in range(5)
} | {
    f"pridict1_2__exp_2023-12-22_16-24-32__run_{fold}"
    for fold in range(5)
}


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
    run_dir = WEIGHTS_ROOT / weight_id
    if weight_id in KNOWN_BROKEN_IDS:
        with pytest.raises(ValueError, match="incomplete"):
            PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))
        return

    PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))


def test_known_broken_bundles_have_mismatched_decoders():
    for weight_id in sorted(KNOWN_BROKEN_IDS):
        run_dir = WEIGHTS_ROOT / weight_id
        if not run_dir.is_dir():
            pytest.skip(f"{weight_id} not present in checkout")

        with open(run_dir / "config" / "exp_options.pkl", "rb") as handle:
            options = pickle.load(handle)
        datasets = options.get("datasets_name", [])
        decoders = sorted(
            path.stem.replace("decoder_", "")
            for path in (run_dir / "model_statedict").glob("decoder_*.pkl")
        )
        assert datasets != decoders

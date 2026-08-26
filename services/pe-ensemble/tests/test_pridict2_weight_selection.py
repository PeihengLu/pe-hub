"""Tests for PRIDICT2 cell-type suffixed weight selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.models.pridict2_wrapper import PRIDICT2ModelWrapper

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_ROOT = REPO_ROOT / "services" / "pe-ensemble" / "weights" / "pridict2"


def _first_loadable_base_id() -> str:
    for entry in PRIDICT2ModelWrapper.list_available_weight_entries():
        weight_id = entry["id"]
        if "__HEK" in weight_id:
            return weight_id.rsplit("__", 1)[0]
    pytest.skip("No loadable PRIDICT2 multi-head weights in checkout")


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_list_available_weight_entries_include_cell_type_suffixes():
    entries = PRIDICT2ModelWrapper.list_available_weight_entries()
    assert entries
    headed = [entry for entry in entries if entry.get("cell_type")]
    assert headed, "expected at least one cell-headed vendor weight"
    assert all("__HEK" in entry["id"] or "__K562" in entry["id"] for entry in headed)
    assert all(entry.get("cell_type") in {"HEK", "K562"} for entry in headed)


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_resolve_weight_selection_parses_cell_type_suffix():
    base_id = _first_loadable_base_id()
    run_dir, cell_type = PRIDICT2ModelWrapper.resolve_weight_selection(f"{base_id}__HEK")
    assert run_dir.is_dir()
    assert cell_type == "HEK"


@pytest.mark.skipif(not WEIGHTS_ROOT.is_dir(), reason="PRIDICT2 weights not available")
def test_resolve_weight_selection_rejects_base_id_for_multi_head_run():
    base_id = _first_loadable_base_id()
    with pytest.raises(ValueError, match="cell-type head suffix"):
        PRIDICT2ModelWrapper.resolve_weight_selection(base_id)


def test_resolve_weight_selection_accepts_single_head_decoder_pkl(tmp_path: Path):
    """Ensemble-trained runs use decoder.pkl; resolve must not require cell heads."""
    import pickle

    run_dir = tmp_path / "smoke_run"
    statedict = run_dir / "model_statedict"
    config = run_dir / "config"
    statedict.mkdir(parents=True)
    config.mkdir(parents=True)
    (statedict / "decoder.pkl").write_bytes(b"x")
    with open(config / "exp_options.pkl", "wb") as handle:
        pickle.dump(
            {
                "model_name": "PE_RNN_distribution",
                "datasets_name": [],
                "separate_attention_layers": False,
                "separate_seqlevel_embedder": False,
            },
            handle,
        )

    resolved, cell = PRIDICT2ModelWrapper.resolve_weight_selection(str(run_dir))
    assert resolved == run_dir.resolve()
    assert cell is None


def test_resolve_weight_selection_rejects_cell_suffix_on_single_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import pickle

    run_dir = tmp_path / "smoke_run"
    statedict = run_dir / "model_statedict"
    config = run_dir / "config"
    statedict.mkdir(parents=True)
    config.mkdir(parents=True)
    (statedict / "decoder.pkl").write_bytes(b"x")
    with open(config / "exp_options.pkl", "wb") as handle:
        pickle.dump(
            {
                "model_name": "PE_RNN_distribution",
                "datasets_name": [],
                "separate_attention_layers": False,
                "separate_seqlevel_embedder": False,
            },
            handle,
        )

    monkeypatch.setattr(
        PRIDICT2ModelWrapper,
        "_split_weight_name",
        staticmethod(lambda name: (str(run_dir), "HEK")),
    )
    with pytest.raises(ValueError, match="single-head"):
        PRIDICT2ModelWrapper.resolve_weight_selection(f"{run_dir}__HEK")


"""Smoke tests guaranteeing all vendor model weights load under one PyTorch.

These tests are the regression guard for the "unified PyTorch environment"
guarantee: DeepPrime, PRIDICT2, and OPED must all load their pre-trained
weights as ``state_dict`` objects in the single torch version pinned by this
service. A future torch bump that silently breaks one model's checkpoint will
fail here.

Each test skips gracefully when the corresponding vendor weights are absent
(e.g. on a checkout without the model submodules populated).
"""
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_MODELS = REPO_ROOT / "vendor" / "models"


def _is_state_dict(obj) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(
        torch.is_tensor(v) for v in obj.values()
    )


def test_torch_is_unified_2x():
    """The environment torch must satisfy the pinned >=2.0,<2.9 range."""
    major, minor = (int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    assert (major, minor) >= (2, 0), f"torch {torch.__version__} < 2.0"
    assert major == 2 and minor < 9, f"torch {torch.__version__} outside <2.9"


def test_deepprime_weights_load_as_state_dict():
    weights = sorted(
        VENDOR_MODELS.glob("deepprime/models/DeepPrime/DeepPrime_base/*.pt")
    )
    if not weights:
        pytest.skip("DeepPrime weights not available")
    sd = torch.load(weights[0], map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)


def test_pridict2_weights_load_as_state_dict():
    components = sorted(
        VENDOR_MODELS.glob(
            "pridict2/trained_models/**/model_statedict/decoder_*.pkl"
        )
    )
    if not components:
        pytest.skip("PRIDICT2 weights not available")
    sd = torch.load(components[0], map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)


def test_oped_weights_load_as_state_dict():
    weights = VENDOR_MODELS.joinpath(
        "oped",
        "pegRNA_PredictingCodes",
        "Model_Trained",
        "pegRNA_Model_Merged_saved.order3_decoder_weights.pt",
    )
    if not weights.is_file():
        pytest.skip("OPED state_dict weights not available")
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)
    # Embeddings encode the per-order k-mer vocab + padding index: [5, 17, 65].
    assert sd["embedding.0.weight"].shape[0] == 5
    assert sd["embedding.1.weight"].shape[0] == 17
    assert sd["embedding.2.weight"].shape[0] == 65


def test_oped_legacy_full_pickle_is_rejected():
    """The legacy full-pickle must not be silently loadable as weights."""
    legacy = VENDOR_MODELS.joinpath(
        "oped",
        "pegRNA_PredictingCodes",
        "Model_Trained",
        "pegRNA_Model_Merged_saved.order3_decoder.pt",
    )
    if not legacy.is_file():
        pytest.skip("OPED legacy full-pickle not present")
    with pytest.raises(Exception):
        torch.load(legacy, map_location="cpu", weights_only=True)

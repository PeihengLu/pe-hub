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
WEIGHTS_ROOT = REPO_ROOT / "services" / "pe-ensemble" / "weights"


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
    registry_weights = sorted(WEIGHTS_ROOT.glob("deepprime/DeepPrime_base/*.pt"))
    vendor_weights = sorted(
        VENDOR_MODELS.glob("deepprime/models/DeepPrime/DeepPrime_base/*.pt")
    )
    weights = registry_weights or vendor_weights
    if not weights:
        pytest.skip("DeepPrime weights not available")
    sd = torch.load(weights[0], map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)


def test_pridict2_weights_load_as_state_dict():
    registry_components = sorted(
        WEIGHTS_ROOT.glob("pridict2/**/model_statedict/decoder_*.pkl")
    )
    vendor_components = sorted(
        VENDOR_MODELS.glob(
            "pridict2/trained_models/**/model_statedict/decoder_*.pkl"
        )
    )
    components = registry_components or vendor_components
    if not components:
        pytest.skip("PRIDICT2 weights not available")
    sd = torch.load(components[0], map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)


def test_oped_weights_load_as_state_dict():
    registry_weights = WEIGHTS_ROOT / "oped" / "pegRNA_Model_Merged_saved.order3_decoder_weights" / "weights.pt"
    vendor_weights = VENDOR_MODELS.joinpath(
        "oped",
        "pegRNA_PredictingCodes",
        "Model_Trained",
        "pegRNA_Model_Merged_saved.order3_decoder_weights.pt",
    )
    weights = registry_weights if registry_weights.is_file() else vendor_weights
    if not weights.is_file():
        pytest.skip("OPED state_dict weights not available")
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    assert _is_state_dict(sd)
    # Embeddings encode the per-order k-mer vocab + padding index: [5, 17, 65].
    assert sd["embedding.0.weight"].shape[0] == 5
    assert sd["embedding.1.weight"].shape[0] == 17
    assert sd["embedding.2.weight"].shape[0] == 65
    # Vendored pretrained checkpoint is the encoder–decoder Order-3 model.
    assert any(key.startswith("encoder_decoder.") for key in sd)


def test_oped_vendor_weights_load_into_model():
    """Regression: decoder checkpoint must load into EncoderDecoder Order-3."""
    registry_dir = WEIGHTS_ROOT / "oped" / "pegRNA_Model_Merged_saved.order3_decoder_weights"
    if not (registry_dir / "weights.pt").is_file():
        pytest.skip("OPED registry weights not available")
    from app.models.oped_wrapper import OPEDModelWrapper

    wrapper = OPEDModelWrapper(device=torch.device("cpu"))
    wrapper.load_weights_by_name("pegRNA_Model_Merged_saved.order3_decoder_weights")
    assert wrapper.is_trained
    assert wrapper.model is not None
    assert type(wrapper.model).__name__ == "TransformerEncoderDecoderModelOrder3"


def test_oped_infer_architecture_pins_nhead_8():
    """Vendor load_model used nhead=64; weight shapes ignore nhead — pin 8."""
    from app.models.oped_wrapper import OPEDModelWrapper

    state = {
        "embedding.0.weight": torch.zeros(5, 64),
        "embedding.1.weight": torch.zeros(17, 64),
        "embedding.2.weight": torch.zeros(65, 64),
        "fully_connected_layers.0.weight": torch.zeros(1, 64),
    }
    arch = OPEDModelWrapper._infer_architecture_from_state_dict(state)
    assert arch["kwargs"]["nhead"] == 8
    assert arch["kwargs"]["embedding_size"] == 64


def test_oped_infer_architecture_falls_back_when_nhead_8_incompatible():
    from app.models.oped_wrapper import OPEDModelWrapper

    state = {
        "embedding.0.weight": torch.zeros(5, 12),
        "embedding.1.weight": torch.zeros(17, 12),
        "embedding.2.weight": torch.zeros(65, 12),
        "fully_connected_layers.0.weight": torch.zeros(1, 12),
    }
    arch = OPEDModelWrapper._infer_architecture_from_state_dict(state)
    assert arch["kwargs"]["nhead"] == 4


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

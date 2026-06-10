"""System test: evaluate() for all PE-ensemble vendor model wrappers.

Uses a tiny fixture dataset under ``testdata/vendor_eval/`` (two pegRNA rows
in each model's native schema). Each test loads bundled pre-trained weights and
runs the wrapper's ``evaluate`` method end-to-end on CPU.

Skips gracefully when vendor weights or optional runtime deps are missing
(same expectation as ``services/pe-ensemble/tests/test_weights_loading.py``).

Run standalone:

    PYTHONPATH=services/pe-ensemble:packages/pe-common \\
        pytest test_vendor_models_evaluation.py -v

Or via ``./run-smoke-tests.sh`` (includes this suite).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
TESTDATA = REPO_ROOT / "testdata" / "vendor_eval"
VENDOR_MODELS = REPO_ROOT / "vendor" / "models"

PE_ENSEMBLE = REPO_ROOT / "services" / "pe-ensemble"
PE_COMMON = REPO_ROOT / "packages" / "pe-common"
for path in (PE_ENSEMBLE, PE_COMMON):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def _model_factory():
    """Import ModelFactory after optional runtime deps are present."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from app.models.model_factory import ModelFactory

    return ModelFactory


def _vendor_available(*parts: str) -> bool:
    return VENDOR_MODELS.joinpath(*parts).exists()


def _require_testdata(filename: str) -> Path:
    path = TESTDATA / filename
    if not path.is_file():
        pytest.skip(f"Fixture not found: {path}")
    return path


def _assert_finite_metric(value: Any, key: str) -> None:
    assert isinstance(value, (int, float)), f"{key} must be numeric, got {type(value)}"
    assert math.isfinite(float(value)), f"{key} must be finite, got {value}"


def _assert_common_metrics(metrics: Dict[str, Any], *, pearson_key: str, spearman_key: str) -> None:
    assert metrics.get("n_samples", 0) >= 2
    _assert_finite_metric(metrics[pearson_key], pearson_key)
    _assert_finite_metric(metrics[spearman_key], spearman_key)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


class TestDeepPrimeEvaluation:
    @pytest.mark.skipif(
        not _vendor_available("deepprime", "models", "DeepPrime"),
        reason="DeepPrime weights not available",
    )
    def test_evaluate_with_bundled_weights(self, device: torch.device) -> None:
        df = pd.read_csv(_require_testdata("deepprime_small.csv"))
        ModelFactory = _model_factory()
        model = ModelFactory.create_model(
            "deepprime",
            device=device,
            pe_system="PE2max",
            cell_type="HEK293T",
        )
        metrics = model.evaluate(df, weights="DeepPrime_base")
        _assert_common_metrics(metrics, pearson_key="pearson", spearman_key="spearman")


class TestOPEDEvaluation:
    @pytest.mark.skipif(
        not _vendor_available(
            "oped",
            "pegRNA_PredictingCodes",
            "Model_Trained",
            "pegRNA_Model_Merged_saved.order3_decoder_weights.pt",
        ),
        reason="OPED weights not available",
    )
    def test_evaluate_with_bundled_weights(self, device: torch.device) -> None:
        native = pd.read_csv(_require_testdata("oped_native_small.csv"))
        labels = pd.read_csv(_require_testdata("standardized_small.csv"))

        ModelFactory = _model_factory()
        model = ModelFactory.create_model("oped", device=device)
        weight_name = model.list_available_weights()[0]
        prepared = model.prepare_data(native)
        prepared["Efficiency"] = labels["editing_efficiency"].astype(float).values

        metrics = model.evaluate(prepared, weights=weight_name)
        _assert_common_metrics(metrics, pearson_key="pearson", spearman_key="spearman")


class TestPRIDICT2Evaluation:
    @pytest.mark.skipif(
        not _vendor_available("pridict2", "trained_models"),
        reason="PRIDICT2 weights not available",
    )
    def test_evaluate_with_bundled_weights(self, device: torch.device) -> None:
        df = pd.read_csv(_require_testdata("pridict2_small.csv"))
        ModelFactory = _model_factory()
        model = ModelFactory.create_model("pridict2", device=device)
        weight_name = next(
            name
            for name in model.list_available_weights()
            if name.endswith("__HEK")
        )

        metrics = model.evaluate(df, weights=weight_name)
        _assert_common_metrics(
            metrics,
            pearson_key="averageedited_pearson",
            spearman_key="averageedited_spearman",
        )


def test_all_vendor_wrappers_listed() -> None:
    """Guard that the system test stays aligned with ModelFactory registrations."""
    ModelFactory = _model_factory()
    assert set(ModelFactory.list_models()) == {"deepprime", "oped", "pridict2"}

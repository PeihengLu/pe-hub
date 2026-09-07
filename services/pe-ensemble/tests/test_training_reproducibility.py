"""Regression tests for training-run reproducibility and checkpoint fidelity.

Each test here pins a property that was silently broken before:

* training runs were unseeded, so an identical request gave different weights;
* a from-scratch DeepPrime checkpoint was saved without its feature
  normalization statistics, so reloading it z-scored inputs with DeepPrime's
  vendor statistics instead;
* ``DeepPrimeModelWrapper.predict`` collapsed to a single scalar whenever the
  ensemble held one model (the from-scratch case);
* missing efficiency labels were imputed as ``0.0`` instead of rejected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pe_common.training import (
    DEFAULT_TRAINING_SEED,
    resolve_training_seed,
)

from app.models import weights_registry
from app.models.deepprime_wrapper import DeepPrimeModelWrapper

# Tabular features DeepPrime's `select_cols` reads, alongside the two 74-mers.
_NUMERIC_FEATURES = (
    "PBSlen", "RTlen", "RT-PBSlen", "Edit_pos", "Edit_len", "RHA_len",
    "Tm1", "Tm2", "Tm2new", "Tm3", "Tm4", "TmD",
    "nGCcnt1", "nGCcnt2", "nGCcnt3", "fGCcont1", "fGCcont2", "fGCcont3",
    "MFE3", "MFE4", "DeepSpCas9_score",
)

_HPARAMS = {"epochs": 2, "batch_size": 16, "hidden_size": 32, "num_layers": 1}


def _deepprime_frame(n_rows: int = 48, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    frame = pd.DataFrame({name: rng.random(n_rows) for name in _NUMERIC_FEATURES})
    frame["WT74_On"] = ["".join(rng.choice(bases, 74)) for _ in range(n_rows)]
    frame["Edited74_On"] = ["".join(rng.choice(bases, 74)) for _ in range(n_rows)]
    frame["type_sub"] = True
    frame["type_ins"] = False
    frame["type_del"] = False
    frame["editing_efficiency"] = rng.random(n_rows) * 40.0
    n_val = max(2, n_rows // 4)
    frame["split"] = ["train"] * (n_rows - n_val) + ["val"] * n_val
    return frame


@pytest.fixture()
def weights_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "weights"
    monkeypatch.setenv("WEIGHTS_ROOT", str(root))
    return root


class TestSeedResolution:
    def test_defaults_to_fixed_seed(self):
        assert resolve_training_seed(None) == DEFAULT_TRAINING_SEED
        assert resolve_training_seed({}) == DEFAULT_TRAINING_SEED

    def test_explicit_seed_wins(self):
        assert resolve_training_seed({"seed": 7}) == 7
        assert resolve_training_seed({"random_state": "11"}) == 11

    @pytest.mark.parametrize("value", [None, "none", ""])
    def test_seeding_can_be_disabled(self, value):
        assert resolve_training_seed({"seed": value}) is None


class TestDeepPrimeReproducibility:
    def test_same_seed_reproduces_training(self):
        frame = _deepprime_frame()

        def run(seed: int) -> float:
            wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
            result = wrapper.train(frame, hyperparameters={**_HPARAMS, "seed": seed})
            return float(result["final_val_loss"])

        assert run(7) == run(7)

    def test_different_seeds_diverge(self):
        frame = _deepprime_frame()

        def run(seed: int) -> float:
            wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
            result = wrapper.train(frame, hyperparameters={**_HPARAMS, "seed": seed})
            return float(result["final_val_loss"])

        assert run(7) != run(99)


class TestDeepPrimeTrainingOutputs:
    def test_train_reports_validation_correlations(self):
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        result = wrapper.train(_deepprime_frame(), hyperparameters=_HPARAMS)
        # The weights registry copies these into the manifest's metrics block.
        assert "validation_metrics" in result
        assert {"pearson", "spearman", "mse", "mae"} <= set(result["validation_metrics"])
        assert "val_pearson" in result and "val_spearman" in result

    def test_from_scratch_training_fits_normalization(self):
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        wrapper.train(_deepprime_frame(), hyperparameters=_HPARAMS)
        assert wrapper.mean is not None and wrapper.std is not None
        # Zero-variance columns must not become a divide-by-zero.
        assert not (wrapper.std == 0).any()

    def test_predict_returns_one_value_per_row(self):
        frame = _deepprime_frame(n_rows=32)
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        wrapper.train(frame, hyperparameters=_HPARAMS)
        # A single-model ensemble used to be squeezed down to one scalar.
        assert len(wrapper.models) == 1
        assert len(wrapper.predict(wrapper.prepare_data(frame))) == len(frame)

    def test_registered_checkpoint_reloads_identically(self, weights_root: Path):
        frame = _deepprime_frame()
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        wrapper.train(frame, hyperparameters=_HPARAMS)
        before = wrapper.predict(wrapper.prepare_data(frame))

        weight_id = weights_registry.register_trained_model(
            "deepprime", wrapper, metadata={"training": {}}
        )
        entry = weights_root / "deepprime" / weight_id
        saved = {path.name for path in entry.iterdir()}
        assert {"mean.csv", "std.csv", "architecture.json"} <= saved

        reloaded = DeepPrimeModelWrapper(device=torch.device("cpu"))
        reloaded.load_weights_by_name(weight_id)
        # Non-default architecture must survive the round trip.
        assert reloaded.architecture["hidden_size"] == _HPARAMS["hidden_size"]
        after = reloaded.predict(reloaded.prepare_data(frame))
        np.testing.assert_allclose(before, after, atol=1e-5)

    def test_saving_without_normalization_is_refused(self, tmp_path: Path):
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        wrapper._init_trainable_models({})
        with pytest.raises(ValueError, match="feature normalization"):
            wrapper.save_model(str(tmp_path / "entry"))


class TestMissingLabelsAreRejected:
    def test_nan_efficiency_raises(self):
        frame = _deepprime_frame()
        frame.loc[0, "editing_efficiency"] = np.nan
        wrapper = DeepPrimeModelWrapper(device=torch.device("cpu"))
        with pytest.raises(ValueError, match="missing/non-numeric"):
            wrapper.train(frame, hyperparameters=_HPARAMS)

"""Guards OptiPrime eval preprocessing against vendor filename parsing."""
from __future__ import annotations

import inspect
from pathlib import Path
import pandas as pd

from pe_ensemble.models.optiprime_wrapper import (
    _PREDICT_CSV_NAME,
    _as_float_scalar,
    _preprocess_optiprime_eval_df,
    OptiPrimeModelWrapper,
)
import numpy as np
import pytest


def test_optiprime_eval_skips_vendor_filename_parser():
    """``process_fname`` rejects non-Liu/Schwank/Kim stems and overwrites metadata."""
    assert _PREDICT_CSV_NAME == "eval.csv"
    predict_src = inspect.getsource(OptiPrimeModelWrapper.predict)
    preprocess_src = inspect.getsource(_preprocess_optiprime_eval_df)
    assert "process_fname" not in predict_src
    assert "process_fname" not in preprocess_src
    assert "_preprocess_optiprime_eval_df" in predict_src
    assert "_PREDICT_CSV_NAME" in predict_src
    assert "_patch_optiprime_scalar_features" in predict_src
    # Fallbacks must use trained group_factor keys, not a synthetic name.
    assert '("group", "Liu_HeLa")' in predict_src
    assert "OptiPrime_HEK293T" not in predict_src
    assert '("cas9_type", "PEmax-Cas9")' in predict_src
    assert '("scaffold_name", "OG_F+E")' in predict_src
    assert '("motif", "tevoPreQ1")' in predict_src


def test_as_float_scalar_accepts_rs3_ndarray():
    assert _as_float_scalar(np.array([0.42])) == pytest.approx(0.42)
    assert _as_float_scalar(np.array([[0.42]])) == pytest.approx(0.42)
    assert _as_float_scalar(0.42) == pytest.approx(0.42)


def test_as_float_scalar_rejects_multi_value():
    with pytest.raises(ValueError, match="expected one value"):
        _as_float_scalar(np.array([0.1, 0.2]))


def test_unknown_weight_does_not_silently_use_base():
    model = OptiPrimeModelWrapper()
    with pytest.raises(ValueError, match="Unknown OptiPrime weights"):
        model.load_weights_by_name("not-an-optiprime-checkpoint")


def test_preprocessing_keeps_unlabeled_and_zero_weight_rows():
    from pe_ensemble.models.optiprime_wrapper import _ensure_optiprime_on_path
    _ensure_optiprime_on_path()
    pytest.importorskip("flax")
    pytest.importorskip("RNA")
    wt = "ACGTGACGTACGTACGTACGTACGTAGGACCTAGCATCGATCGTAGC"
    df = pd.DataFrame({"spacer": [wt[4:24]] * 3,
                       "rtt": ["ACGTACGTACGT"] * 3,
                       "pbs": ["ACGTACGT"] * 3,
                       "full_unedited": [wt] * 3,
                       "full_edited": [wt[:30] + "A" + wt[31:]] * 3,
                       "edited_frac": [0.2, np.nan, 0.3],
                       "indel_frac": [0.0, np.nan, 0.0],
                       "weight": [1.0, 1.0, 0.0],
                       "time": [3.0, 5.0, 7.0],
                       "group": ["Liu_HEK293T", "Liu_HeLa", "Kim_HEK293T"]})
    result = _preprocess_optiprime_eval_df(Path("eval.csv"), df)
    assert len(result) == len(df)
    assert result["group"].tolist() == df["group"].tolist()
    assert result["time"].tolist() == [2.0, 4.0, 6.0]
    assert df["weight"].tolist() == [1.0, 1.0, 0.0]

"""Guards OptiPrime eval preprocessing against vendor filename parsing."""
from __future__ import annotations

import inspect

from app.models.optiprime_wrapper import (
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


def test_as_float_scalar_accepts_rs3_ndarray():
    assert _as_float_scalar(np.array([0.42])) == pytest.approx(0.42)
    assert _as_float_scalar(np.array([[0.42]])) == pytest.approx(0.42)
    assert _as_float_scalar(0.42) == pytest.approx(0.42)


def test_as_float_scalar_rejects_multi_value():
    with pytest.raises(ValueError, match="expected one value"):
        _as_float_scalar(np.array([0.1, 0.2]))

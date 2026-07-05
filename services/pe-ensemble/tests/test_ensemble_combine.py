import numpy as np
import pytest

from app.ensemble.combine import combine_predictions, validate_combine_method


def test_mean_combine():
    preds = np.array([[0.2, 0.4], [0.6, 0.8]])
    combined = combine_predictions(preds, "mean")
    np.testing.assert_allclose(combined, [0.3, 0.7])


def test_weighted_mean_combine():
    preds = np.array([[0.0, 1.0], [0.2, 0.8]])
    combined = combine_predictions(preds, "weighted_mean", options={"weights": [0.25, 0.75]})
    np.testing.assert_allclose(combined, [0.75, 0.65])


def test_median_and_trimmed_mean():
    preds = np.array([[0.1, 0.5, 0.9], [0.0, 0.4, 1.0]])
    np.testing.assert_allclose(combine_predictions(preds, "median"), [0.5, 0.4])
    np.testing.assert_allclose(
        combine_predictions(preds, "trimmed_mean", options={"trim_count": 1}),
        [0.5, 0.4],
    )


def test_rank_and_percentile_mean_shapes():
    preds = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
        ]
    )
    rank_combined = combine_predictions(preds, "rank_mean")
    percentile_combined = combine_predictions(preds, "percentile_mean")
    assert rank_combined.shape == (4,)
    assert percentile_combined.shape == (4,)
    assert np.all(np.isfinite(rank_combined))
    assert np.all(np.isfinite(percentile_combined))


def test_geometric_and_harmonic_mean():
    preds = np.array([[0.2, 0.5], [0.4, 0.8]])
    geometric = combine_predictions(preds, "geometric_mean")
    harmonic = combine_predictions(preds, "harmonic_mean")
    assert np.all(geometric > 0)
    assert np.all(harmonic > 0)
    assert np.all(geometric <= combine_predictions(preds, "mean"))
    assert np.all(harmonic <= geometric)


def test_min_max():
    preds = np.array([[0.2, 0.8], [0.1, 0.9]])
    np.testing.assert_allclose(combine_predictions(preds, "min"), [0.2, 0.1])
    np.testing.assert_allclose(combine_predictions(preds, "max"), [0.8, 0.9])


def test_validate_combine_method_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid combine method"):
        validate_combine_method("stacking")

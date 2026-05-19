import numpy as np
import pytest

from benchmark_utils.predictions import cutoff_scores


def test_cutoff_scores_returns_none_without_cutoff():
    scores = np.array([0.1, 0.8, 0.2])

    assert cutoff_scores(scores) is None


def test_cutoff_scores_uses_top_score_fraction():
    scores = np.array([0.1, 0.8, 0.2, 0.9])

    predictions = cutoff_scores(scores, cutoff=0.25)

    np.testing.assert_array_equal(predictions, np.array([0, 0, 0, 1]))


def test_cutoff_scores_preserves_nan_padding_as_ignore_label():
    scores = np.array([np.nan, 0.1, 0.8, 0.2, 0.9])

    predictions = cutoff_scores(scores, cutoff=0.25)

    np.testing.assert_array_equal(predictions, np.array([-1, 0, 0, 0, 1]))


def test_cutoff_scores_rejects_invalid_cutoff():
    scores = np.array([0.1, 0.8, 0.2])

    with pytest.raises(ValueError, match="must be in"):
        cutoff_scores(scores, cutoff=1)

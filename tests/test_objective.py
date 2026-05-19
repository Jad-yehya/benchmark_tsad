import numpy as np
import pytest

from objective import Objective


def make_objective(score_metrics=("auc_pr", "auc_roc"),
                   prediction_metrics=None):
    objective = Objective()
    objective.score_metrics = score_metrics
    objective.prediction_metrics = prediction_metrics
    objective.set_data(
        X_train=np.empty((1, 1, 6)),
        y_test=np.array([0, 0, 1, 0, 1, 0]),
        X_test=np.empty((1, 1, 6)),
    )
    return objective


def test_default_evaluation_uses_score_metrics_only():
    objective = make_objective()
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.2])

    result = objective.evaluate_result(anomaly_scores=scores)

    assert result["auc_pr"] == pytest.approx(1.0)
    assert result["auc_roc"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(0.0)
    assert "precision" not in result


def test_score_and_prediction_metrics_use_canonical_keys():
    objective = make_objective(
        score_metrics=("auc_pr",),
        prediction_metrics=("precision",),
    )
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.2])
    predictions = np.array([0, 0, 1, 0, 1, 0])

    result = objective.evaluate_result(
        anomaly_scores=scores,
        anomaly_predictions=predictions,
    )

    assert result["auc_pr"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)


def test_prediction_metrics_are_opt_in():
    objective = make_objective(
        prediction_metrics=("precision", "recall", "f1", "zoloss"),
    )
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.2])
    predictions = np.array([0, 0, 1, 0, 1, 0])

    result = objective.evaluate_result(
        anomaly_scores=scores,
        anomaly_predictions=predictions,
    )

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["zoloss"] == pytest.approx(0.0)


def test_prediction_metrics_require_prediction_array():
    objective = make_objective(prediction_metrics=("precision",))
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.8, 0.2])

    with pytest.raises(ValueError, match="anomaly_predictions"):
        objective.evaluate_result(anomaly_scores=scores)


def test_nan_score_padding_is_masked():
    objective = make_objective()
    scores = np.array([np.nan, 0.2, 0.9, 0.1, 0.8, 0.2])

    result = objective.evaluate_result(anomaly_scores=scores)

    assert result["auc_pr"] == pytest.approx(1.0)
    assert result["auc_roc"] == pytest.approx(1.0)


def test_prediction_padding_is_masked():
    objective = make_objective(
        score_metrics=None,
        prediction_metrics=("precision", "recall", "f1"),
    )
    predictions = np.array([-1, 0, 1, 0, 1, 0])

    result = objective.evaluate_result(anomaly_predictions=predictions)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(0.0)


def test_prediction_only_metrics_without_primary_value_fallback_to_zero():
    objective = make_objective(
        score_metrics=None,
        prediction_metrics=("precision",),
    )
    predictions = np.array([0, 0, 1, 0, 1, 0])

    result = objective.evaluate_result(anomaly_predictions=predictions)

    assert result["precision"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(0.0)

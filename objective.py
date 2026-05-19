from benchopt import BaseObjective
from benchmark_utils.metrics import (
    soft_precision as soft_precision_score,
    soft_recall as soft_recall_score,
    soft_f1 as soft_f1_score,
    ctt,
    ttc,
    extract_anomaly_ranges,
    precision_t as precision_t_score,
    recall_t as recall_t_score,
    f1_t as f1_t_score,
)

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    zero_one_loss,
    roc_auc_score,
)


class Objective(BaseObjective):
    name = "Anomaly detection"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "score_metrics": [("auc_pr", "auc_roc")],
        "prediction_metrics": [None],
    }

    detection_ranges = (1, 3, 5, 10, 20)
    default_prediction_metrics = (
        "precision",
        "recall",
        "f1",
        "precision_t",
        "recall_t",
        "f1_t",
        "ctt",
        "ttc",
        "zoloss",
        "soft_precision",
        "soft_recall",
        "soft_f1",
    )

    def get_one_result(self):
        """Return one solution for which the objective can be computed."""
        score_metrics = self._normalize_metrics(
            getattr(self, "score_metrics", ("auc_pr", "auc_roc"))
        )
        prediction_metrics = self._expand_prediction_metrics(
            getattr(self, "prediction_metrics", None)
        )

        result = {}
        if score_metrics:
            result["anomaly_scores"] = np.zeros_like(
                self.y_test, dtype=float
            )
        if prediction_metrics:
            result["anomaly_predictions"] = np.zeros_like(
                self.y_test, dtype=int
            )
        return result

    def set_data(self, X_train, y_test, X_test):
        "Set the data to compute the objective."
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def evaluate_result(
        self,
        anomaly_scores=None,
        anomaly_predictions=None,
    ):
        """Evaluate the result provided by the solver.

        anomaly_scores is the score-based solver output.
        anomaly_predictions is optional and only needed when requesting
        prediction-based metrics.
        """
        score_metrics = self._normalize_metrics(
            getattr(self, "score_metrics", ("auc_pr", "auc_roc"))
        )
        prediction_metrics = self._expand_prediction_metrics(
            getattr(self, "prediction_metrics", None)
        )

        if score_metrics and anomaly_scores is None:
            raise ValueError("score_metrics require an anomaly_scores array.")
        if prediction_metrics and anomaly_predictions is None:
            raise ValueError(
                "prediction_metrics require an anomaly_predictions array.")

        y_true, scores, predictions = self._align_inputs(
            anomaly_scores=anomaly_scores,
            anomaly_predictions=anomaly_predictions,
        )

        result = {}
        if score_metrics:
            result.update(
                self._compute_score_metrics(
                    y_true=y_true,
                    anomaly_scores=scores,
                    metrics=score_metrics,
                )
            )
        if prediction_metrics:
            result.update(
                self._compute_prediction_metrics(
                    y_true=y_true,
                    anomaly_predictions=predictions,
                    metrics=prediction_metrics,
                )
            )

        # Setting value to 0. The actual value is not used for ranking.
        result["value"] = 0.0
        return result

    def get_objective(self):
        return dict(X_train=self.X_train, X_test=self.X_test)

    def _normalize_metrics(self, metrics):
        if metrics is None:
            return ()
        if isinstance(metrics, str):
            if metrics == "all":
                return ("auc_pr", "auc_roc")
            return (metrics,)
        return tuple(metric for metric in metrics if metric is not None)

    def _expand_prediction_metrics(self, metrics):
        metrics = self._normalize_prediction_metrics(metrics)
        expanded = []

        for metric in metrics:
            if metric == "all":
                metric = self.default_prediction_metrics
            else:
                metric = (metric,)

            for name in metric:
                if name in {
                    "soft_precision",
                    "soft_recall",
                    "soft_f1",
                }:
                    expanded.extend(
                        f"{name}_{detection_range}"
                        for detection_range in self.detection_ranges
                    )
                else:
                    expanded.append(name)

        return tuple(expanded)

    def _normalize_prediction_metrics(self, metrics):
        if metrics is None:
            return ()
        if isinstance(metrics, str):
            return (metrics,)
        return tuple(metric for metric in metrics if metric is not None)

    def _align_inputs(self, anomaly_scores, anomaly_predictions):
        # flatten everything before aligning lengths.
        y_true = np.asarray(self.y_test).reshape(-1)
        scores = self._as_flat_array(anomaly_scores)
        predictions = self._as_flat_array(anomaly_predictions)

        # Only align against arrays that were returned. This keeps
        # score-only and prediction-only evaluations valid.
        arrays = [array for array in (
            scores, predictions) if array is not None]
        if not arrays:
            return y_true, None, None

        # Windowed solvers return fewer outputs than y_test because the
        # first timestamps have no full context window. Keep the last samples,
        # which correspond to the part of y_test the solver scored.
        length = min([len(y_true)] + [len(array) for array in arrays])
        y_true = y_true[-length:]
        if scores is not None:
            scores = scores[-length:]
        if predictions is not None:
            predictions = predictions[-length:]

        # Drop invalid positions. NaN score padding and -1 prediction padding
        # When both scores and predictions are present, the same mask is
        # applied to keep mixed metric requests on the same timestamps.
        valid = np.ones(length, dtype=bool)
        if scores is not None:
            valid &= ~np.isnan(scores)
        if predictions is not None:
            valid &= ~np.isnan(predictions)
            valid &= predictions != -1

        y_true = y_true[valid]
        if scores is not None:
            scores = scores[valid]
        if predictions is not None:
            predictions = predictions[valid]

        return y_true, scores, predictions

    def _as_flat_array(self, array):
        if array is None:
            return None
        return np.asarray(array).reshape(-1)

    def _compute_score_metrics(self, y_true, anomaly_scores, metrics):
        if len(y_true) == 0:
            return {metric: np.nan for metric in metrics}

        result = {}
        for metric in metrics:
            if metric == "auc_roc":
                result[metric] = self._safe_auc_roc(y_true, anomaly_scores)
            elif metric == "auc_pr":
                result[metric] = self._auc_pr(y_true, anomaly_scores)
            else:
                raise ValueError(f"Unknown score metric: {metric}")
        return result

    def _compute_prediction_metrics(
            self,
            y_true,
            anomaly_predictions,
            metrics,
    ):
        if len(y_true) == 0:
            return {metric: np.nan for metric in metrics}

        result = {}
        anomaly_ranges = None
        prediction_ranges = None

        for metric in metrics:
            if metric == "precision":
                result[metric] = precision_score(
                    y_true, anomaly_predictions, zero_division=0
                )
            elif metric == "recall":
                result[metric] = recall_score(
                    y_true, anomaly_predictions, zero_division=0
                )
            elif metric == "f1":
                result[metric] = f1_score(
                    y_true, anomaly_predictions, zero_division=0)
            elif metric == "zoloss":
                result[metric] = zero_one_loss(y_true, anomaly_predictions)
            elif metric in {"precision_t", "recall_t", "f1_t"}:
                if anomaly_ranges is None:
                    anomaly_ranges, prediction_ranges = self._get_ranges(
                        y_true, anomaly_predictions
                    )
                if metric == "precision_t":
                    result[metric] = precision_t_score(
                        anomaly_ranges, prediction_ranges
                    )
                elif metric == "recall_t":
                    result[metric] = recall_t_score(
                        anomaly_ranges, prediction_ranges)
                else:
                    result[metric] = f1_t_score(
                        anomaly_ranges, prediction_ranges)
            elif metric == "ctt":
                result[metric] = ctt(y_true, anomaly_predictions)
            elif metric == "ttc":
                result[metric] = ttc(y_true, anomaly_predictions)
            elif metric.startswith("soft_precision_"):
                detection_range = self._parse_detection_range(
                    metric, "soft_precision")
                result[metric] = soft_precision_score(
                    y_true,
                    anomaly_predictions,
                    detection_range=detection_range,
                )
            elif metric.startswith("soft_recall_"):
                detection_range = self._parse_detection_range(
                    metric, "soft_recall")
                result[metric] = soft_recall_score(
                    y_true,
                    anomaly_predictions,
                    detection_range=detection_range,
                )
            elif metric.startswith("soft_f1_"):
                detection_range = self._parse_detection_range(
                    metric, "soft_f1")
                result[metric] = soft_f1_score(
                    y_true,
                    anomaly_predictions,
                    detection_range=detection_range,
                )
            else:
                raise ValueError(f"Unknown prediction metric: {metric}")

        return result

    def _get_ranges(self, y_true, anomaly_predictions):
        return (
            extract_anomaly_ranges(y_true),
            extract_anomaly_ranges(anomaly_predictions),
        )

    def _parse_detection_range(self, metric, prefix):
        suffix = metric.replace(f"{prefix}_", "", 1)
        try:
            return int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"Invalid detection range in prediction metric: {metric}"
            ) from exc

    def _safe_auc_roc(self, y_true, anomaly_scores):
        return roc_auc_score(y_true, anomaly_scores)

    def _auc_pr(self, y_true, anomaly_scores):
        if len(np.unique(y_true)) == 1:
            return np.nan
        return average_precision_score(y_true, anomaly_scores)

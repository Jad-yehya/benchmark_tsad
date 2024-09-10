from benchopt import BaseObjective, safe_import_context
from benchmark_utils.metrics import (
    soft_precision as soft_precision_score,
    soft_recall as soft_recall_score,
    soft_f1 as soft_f1_score,
    ctt, ttc,
    extract_anomaly_ranges,
    precision_t as precision_t_score,
    recall_t as recall_t_score,
    f1_t as f1_t_score
)

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, zero_one_loss
    )


class Objective(BaseObjective):
    name = "Anomaly detection"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def get_one_result(self):
        """Return one solution for which the objective can be computed,
        Used to get the shape of the result.
        Our algorithms will return an array of labels of shape (n_samples,)
        """
        return dict(y_hat=np.ones(self.X_test.shape[0]))

    def set_data(self, X_train, y_test, X_test):
        "Set the data to compute the objective."
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

    def evaluate_result(self, y_hat):
        """Evaluate the result provided by the solver."""
        to_discard = (y_hat == -1).sum()
        self.y_test = self.y_test[to_discard:]
        y_hat = y_hat[to_discard:]

        result = {}
        detection_ranges = [1, 3, 5, 10, 20]

        # Standard metrics
        precision = precision_score(self.y_test, y_hat, zero_division=0)
        recall = recall_score(self.y_test, y_hat, zero_division=0)
        f1 = f1_score(self.y_test, y_hat, zero_division=0)

        anomaly_ranges = extract_anomaly_ranges(self.y_test)
        prediction_ranges = extract_anomaly_ranges(y_hat)

        precision_t = precision_t_score(anomaly_ranges, prediction_ranges)
        recall_t = recall_t_score(anomaly_ranges, prediction_ranges)
        f1_t = f1_t_score(anomaly_ranges, prediction_ranges)

        result.update({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        for range_value in detection_ranges:
            soft_precision = soft_precision_score(
                self.y_test, y_hat, detection_range=range_value
            )
            soft_recall = soft_recall_score(
                self.y_test, y_hat, detection_range=range_value
            )
            soft_f1 = soft_f1_score(soft_precision, soft_recall)

            result.update({
                f"soft_precision_{range_value}": soft_precision,
                f"soft_recall_{range_value}": soft_recall,
                f"soft_f1_{range_value}": soft_f1
            })

        zoloss = zero_one_loss(self.y_test, y_hat)

        # Other metrics
        cct_score = ctt(self.y_test, y_hat)
        ttc_score = ttc(self.y_test, y_hat)

        # Add remaining metrics to the result dictionary
        result.update({
            "precision_t": precision_t,
            "recall_t": recall_t,
            "f1_t": f1_t,
            "cct": cct_score,
            "ttc": ttc_score,
            "zoloss": zoloss,
            "value": zoloss  # having zoloss twice for the API
        })

        return result

    def get_objective(self):
        return dict(
            X_train=self.X_train, y_test=None, X_test=self.X_test
        )

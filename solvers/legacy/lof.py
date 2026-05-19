# Local Outlier Factor

from benchopt import BaseSolver

from sklearn.neighbors import LocalOutlierFactor
import numpy as np

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "LocalOutlierFactor"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [0.1, 0.2, 0.3],
        "n_neighbors": [5, 10, 20, 25, 40],
        "window": [True],
        "window_size": [20],
        "stride": [1],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.clf = LocalOutlierFactor(
            novelty=True,
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )

    def run(self, _):
        if self.window:
            # We need to transform the data to have a rolling window
            if self.X_train is not None:
                self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            if self.X_test is not None:
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            flatrain = self.Xw_train.reshape(self.Xw_train.shape[0], -1)
            flatest = self.Xw_test.reshape(self.Xw_test.shape[0], -1)

            self.clf.fit(flatrain)
            anomaly_scores = -self.clf.decision_function(flatest)

            # Anomaly scores
            self.anomaly_scores = np.array(anomaly_scores)
            padding = max(self.X_test.shape[0] - len(self.anomaly_scores), 0)
            self.anomaly_scores = np.append(
                np.full(padding, np.nan),
                self.anomaly_scores,
            )
            self.anomaly_predictions = cutoff_scores(
                self.anomaly_scores,
                cutoff=self.cutoff,
            )

    def skip(self, X_train, X_test):
        if self.n_neighbors > self.window_size:
            return True, "Number of neighbors greater than number of samples."
        if self.n_neighbors > X_train.shape[0]:
            return True, "Number of neighbors greater than number of samples."
        if X_train.shape[0] < self.window_size:
            return True, "No enough samples to create a window"
        return False, None

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        # To ignore : -1
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

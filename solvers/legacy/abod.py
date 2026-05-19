# ABOD solver

from benchopt import BaseSolver

from pyod.models.abod import ABOD
import numpy as np

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "ABOD"  # Angle-Based Outlier Detection

    install_cmd = "conda"
    requirements = ["pip::pyod"]

    parameters = {
        "contamination": [5e-4, 0.1, 0.2, 0.3],
        "n_neighbors": [5, 10, 15, 20, 30],
        "window": [True],
        "window_size": [20],
        "stride": [1],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.clf = ABOD(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            method="fast"
        )

    def run(self, _):
        # Using only windowed data, parameter used only for consistency
        if self.window:

            # Transofrming the data into rolling windowed data
            if self.X_train is not None:
                self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            if self.X_test is not None:
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            # Flattening the data for the model
            flatrain = self.Xw_train.reshape(self.Xw_train.shape[0], -1)
            flatest = self.Xw_test.reshape(self.Xw_test.shape[0], -1)

            self.clf.fit(flatrain)
            anomaly_scores = self.clf.decision_function(flatest)

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

    # Function used to skip a solver call when n_neighbors >= window_size
    def skip(self, X_train, X_test):
        if self.n_neighbors >= self.window_size:
            return True, "Number of neighbors greater than number of samples."
        return False, None

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        # To ignore : -1
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

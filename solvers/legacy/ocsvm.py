from benchopt import BaseSolver

from sklearn.svm import OneClassSVM
import numpy as np

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "OCSVM"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "nu": [0.001, 0.01, 0.05],
        "gamma": [1e-5, 1e-2],
        "kernel": ["rbf"],
        "window": [True],
        "window_size": [128],
        "stride": [1],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.clf = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )

        if self.window:
            if self.X_train is not None:
                self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            if self.X_test is not None:
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

            self.flatrain = self.Xw_train.reshape(self.Xw_train.shape[0], -1)
            self.flatest = self.Xw_test.reshape(self.Xw_test.shape[0], -1)

    def run(self, _):
        if self.window:
            self.clf.fit(self.flatrain)
            anomaly_scores = -self.clf.decision_function(self.flatest)

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
        if X_train.shape[0] < self.window_size:
            return True, "Window size is larger than dataset size."
        return False, None

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

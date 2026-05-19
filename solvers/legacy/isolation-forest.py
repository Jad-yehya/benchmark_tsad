# Isolation Forest solver

from benchopt import BaseSolver

from sklearn.ensemble import IsolationForest
import numpy as np

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "IsolationForest"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [5e-4, 5e-3, 5e-2, 0.1, 0.2, 0.4, 0.5],
        "window": [True],
        "window_size": [60, 120, 180],
        "stride": [1],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        n_recordings, n_features, n_samples = self.X_train.shape
        self.clf = IsolationForest(contamination=self.contamination)

    def run(self, _):
        if self.window:
            # We need to transform the data to have a rolling window
            if self.X_train is not None:
                # Apply sliding window along the time dimension (axis=2)
                n_recordings, n_features, n_samples = self.X_train.shape
                self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=2
                )[:, :, ::self.stride].transpose(0, 1, 3, 2)

            if self.X_test is not None:
                n_recordings, n_features, n_samples = self.X_test.shape
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=2
                )[:, :, ::self.stride].transpose(0, 1, 3, 2)

            # Flatten for sklearn
            flatrain = self.Xw_train.reshape(
                self.Xw_train.shape[0] * self.Xw_train.shape[1], -1)
            flatest = self.Xw_test.reshape(
                self.Xw_test.shape[0] * self.Xw_test.shape[1], -1)

            self.clf.fit(flatrain)
            anomaly_scores = -self.clf.decision_function(flatest)

            # The results we get has a shape of
            n_recordings, n_features, n_windows, _ = self.Xw_test.shape

            # Anomaly scores
            self.anomaly_scores = np.array(anomaly_scores)
            self.anomaly_scores = self.anomaly_scores.reshape(
                n_recordings, n_features, n_windows)
        else:
            # No windowing case
            # Flatten the data for sklearn
            n_recordings, n_features, n_samples = self.X_train.shape
            X_train_flat = self.X_train.reshape(-1, n_features)
            X_test_flat = self.X_test.reshape(-1, n_features)

            self.clf.fit(X_train_flat)
            self.anomaly_scores = -self.clf.decision_function(X_test_flat)

            # Reshape to (n_recordings, n_samples) for single feature case
            # We assume we take the first feature or average across features
            self.anomaly_scores = self.anomaly_scores.reshape(
                n_recordings, n_samples)

        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def skip(self, X_train, X_test):
        # Skip if dataset size is smaller than window size
        _, _, n_samples = X_train.shape
        if n_samples < self.window_size:
            return True, "Window size is larger than dataset size. Skipping."
        return False, None

    def get_result(self):
        # Anomaly : 1
        # Inlier : 0
        # To ignore : -1
        # For now, take the first recording
        anomaly_scores = self.anomaly_scores[0] if (
            self.anomaly_scores.ndim > 1
        ) else self.anomaly_scores
        result = dict(anomaly_scores=anomaly_scores)
        if self.anomaly_predictions is not None:
            anomaly_predictions = self.anomaly_predictions[0] if (
                self.anomaly_predictions.ndim > 1
            ) else self.anomaly_predictions
            result["anomaly_predictions"] = anomaly_predictions
        return result

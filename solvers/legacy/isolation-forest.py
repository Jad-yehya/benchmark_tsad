# Isolation Forest solver

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.ensemble import IsolationForest
    import numpy as np


class Solver(BaseSolver):
    name = "IsolationForest"

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "contamination": [5e-4, 5e-3, 5e-2, 0.1, 0.2, 0.4, 0.5],
        "window": [True],
        "window_size": [60, 120, 180],
        "stride": [1],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
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

            if self.y_test is not None:
                n_recordings, _, n_samples = self.y_test.shape
                self.yw_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=2
                )[:, :, ::self.stride]

            # Flatten for sklearn
            flatrain = self.Xw_train.reshape(
                self.Xw_train.shape[0] * self.Xw_train.shape[1], -1)
            flatest = self.Xw_test.reshape(
                self.Xw_test.shape[0] * self.Xw_test.shape[1], -1)

            self.clf.fit(flatrain)
            raw_y_hat = self.clf.predict(flatest)
            raw_anomaly_score = self.clf.decision_function(flatest)

            # The results we get has a shape of
            n_recordings, n_features, n_windows, _ = self.Xw_test.shape

            # Mapping the binary output from {-1, 1} to {1, 0}
            # For consistency with the other solvers
            self.raw_y_hat = np.array(raw_y_hat)
            self.raw_y_hat = np.where(self.raw_y_hat == -1, 1, 0)

            # Reshape back to original structure
            self.raw_y_hat = self.raw_y_hat.reshape(
                n_recordings, n_features, n_windows)

            # Anomaly scores (Not used but allows finer thresholding)
            self.raw_anomaly_score = np.array(raw_anomaly_score)
            self.raw_anomaly_score = self.raw_anomaly_score.reshape(
                n_recordings, n_features, n_windows)
        else:
            # No windowing case
            # Flatten the data for sklearn
            n_recordings, n_features, n_samples = self.X_train.shape
            X_train_flat = self.X_train.reshape(-1, n_features)
            X_test_flat = self.X_test.reshape(-1, n_features)

            self.clf.fit(X_train_flat)
            self.raw_y_hat = self.clf.predict(X_test_flat)
            self.raw_anomaly_score = self.clf.decision_function(X_test_flat)

            # Reshape to (n_recordings, n_samples) for single feature case
            # We assume we take the first feature or average across features
            self.raw_y_hat = self.raw_y_hat.reshape(n_recordings, n_samples)
            self.raw_anomaly_score = self.raw_anomaly_score.reshape(
                n_recordings, n_samples)

    def skip(self, X_train, X_test, y_test):
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
        self.y_hat = self.raw_y_hat[0] if (
            self.raw_y_hat.ndim > 1
        ) else self.raw_y_hat
        return dict(y_hat=self.y_hat)

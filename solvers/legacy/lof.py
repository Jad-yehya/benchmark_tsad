# Local Outlier Factor

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.neighbors import LocalOutlierFactor
    import numpy as np


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
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
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

            if self.y_test is not None:
                self.yw_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=0
                )[::self.stride]

            flatrain = self.Xw_train.reshape(self.Xw_train.shape[0], -1)
            flatest = self.Xw_test.reshape(self.Xw_test.shape[0], -1)

            self.clf.fit(flatrain)
            raw_y_hat = self.clf.predict(flatest)
            raw_anomaly_score = self.clf.decision_function(flatest)

            # The results we get has a shape of
            result_shape = (
                (self.X_train.shape[0] - self.window_size) // self.stride
            ) + 1

            # Mapping the binary output from {-1, 1} to {1, 0}
            # For consistency with the other solvers
            self.raw_y_hat = np.array(raw_y_hat)
            self.raw_y_hat = np.where(self.raw_y_hat == -1, 1, 0)

            # Adding -1 for the non predicted samples
            # The first window_size samples are not predicted by the model
            self.raw_y_hat = np.append(
                np.full(self.X_train.shape[0] -
                        result_shape, -1), self.raw_y_hat
            )

            # Anomaly scores (Not used but allows finer thresholding)
            self.raw_anomaly_score = np.array(raw_anomaly_score)
            self.raw_anomaly_score = np.append(
                np.full(result_shape, -1), self.raw_anomaly_score
            )

    def skip(self, X_train, y_test, X_test):
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
        self.y_hat = self.raw_y_hat
        return dict(y_hat=self.y_hat)

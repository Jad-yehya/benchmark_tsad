from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.svm import OneClassSVM
    import numpy as np


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
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
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

            if self.y_test is not None:
                self.yw_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=0
                )[::self.stride]

            self.flatrain = self.Xw_train.reshape(self.Xw_train.shape[0], -1)
            self.flatest = self.Xw_test.reshape(self.Xw_test.shape[0], -1)

    def run(self, _):
        if self.window:
            self.clf.fit(self.flatrain)
            raw_y_hat = self.clf.predict(self.flatest)
            raw_anomaly_score = self.clf.decision_function(self.flatest)

            # The results we get has a shape of
            result_shape = (
                (self.X_train.shape[0] - self.window_size) // self.stride
            ) + 1

            # Mapping the binary output from {-1, 1} to {1, 0}
            # For consistency with the other solvers
            self.raw_y_hat = np.array(raw_y_hat)

            # Adding -1 for the non predicted samples
            # The first window_size samples are not predicted by the model
            self.raw_y_hat = np.where(self.raw_y_hat == -1, 1, 0)
            self.raw_y_hat = np.append(
                np.full(self.X_train.shape[0] -
                        result_shape, -1), self.raw_y_hat
            )

            # Anomaly scores (Not used but allows finer thresholding)
            self.raw_anomaly_score = np.array(raw_anomaly_score)
            self.raw_anomaly_score = np.append(
                np.full(result_shape, -1), self.raw_anomaly_score
            )

    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0] < self.window_size:
            return True, "Window size is larger than dataset size."
        return False, None

    def get_result(self):
        return dict(y_hat=self.raw_y_hat)

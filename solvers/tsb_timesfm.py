from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from TSB_AD.model_wrapper import run_TimesFM
    import numpy as np


class Solver(BaseSolver):
    name = "TSB-TimesFM"

    install_cmd = "conda"
    requirements = ["pip:tsb-ad"]

    parameters = {
        "win_size": [96],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        _, n_features, _ = X_train.shape
        self.data = np.append(X_train, X_test, axis=2)
        self.data = self.data.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

    def run(self, _):
        self.y_hat = run_TimesFM(
            data=self.data,
            win_size=self.win_size,
        )
        self.raw_anomaly_score = self.y_hat[-len(self.X_test):]

    def get_result(self):
        threshold = np.percentile(self.raw_anomaly_score, 90)
        self.y_hat = (self.raw_anomaly_score > threshold).astype(int)
        return dict(y_hat=self.y_hat, raw_anomaly_score=self.raw_anomaly_score)

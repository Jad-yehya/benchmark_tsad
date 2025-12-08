from benchopt import BaseSolver

import torch
import numpy as np
from TSB_AD.model_wrapper import run_TimesFM


class Solver(BaseSolver):
    name = "TSB-TimesFM"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad", "pip::timesfm"]

    parameters = {
        "win_size": [256],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
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
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        threshold = np.percentile(self.raw_anomaly_score, 90)
        self.y_hat = (self.raw_anomaly_score > threshold).astype(int)
        return dict(y_hat=self.y_hat, raw_anomaly_score=self.raw_anomaly_score)

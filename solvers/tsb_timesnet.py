from benchopt import BaseSolver

import torch
from TSB_AD.models.TimesNet import TimesNet

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "TSB-TimesNet"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad"]

    parameters = {
        "window_size": [256],
        "lr": [1e-4],
        "epochs": [10],
        "batch_size": [128],
        "cutoff": [None],
    }

    test_config = {
        "dataset": {
            "n_samples": 512,
            "n_features": 2,
            "n_anomaly": 32,
        },
        "window_size": 32,
        "epochs": 1,
        "batch_size": 16,
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        _, n_features, _ = X_train.shape
        self.X_train = X_train.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

        self.clf = TimesNet(
            win_size=self.window_size,
            enc_in=n_features,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=3,
            features="M",
            lradj="type1",
            validation_size=0.2,
        )

    def run(self, _):
        self.clf.fit(self.X_train)
        self.anomaly_scores = self.clf.decision_function(self.X_test)
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

        del self.clf.model
        del self.clf
        torch.cuda.empty_cache()  # Release cached GPU memory

    def skip(self, X_train, X_test):
        if X_train.shape[-1] < self.window_size:
            return True, "Not enough training samples to create a window."
        if X_test.shape[-1] < self.window_size:
            return True, "Not enough testing samples to create a window."
        return False, None

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

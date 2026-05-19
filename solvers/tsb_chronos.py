from benchopt import BaseSolver

import torch
import numpy as np
from TSB_AD.models.Chronos import Chronos
from TSB_AD.utils.slidingWindows import find_length

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "TSB-Chronos"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad"]

    parameters = {
        "win_size": ["auto"],
        "prediction_length": [1],
        "model_size": ['base'],
        "batch_size": [32],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        _, n_features, _ = X_train.shape
        self.data = np.append(X_train, X_test, axis=2)
        self.data = self.data.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

        if self.win_size == "auto":
            self.win_size = int(find_length(X_train.reshape(-1)))

        self.clf = Chronos(
            win_size=self.win_size,
            input_c=n_features,
            prediction_length=self.prediction_length,
            model_size=self.model_size,
            batch_size=self.batch_size,
        )

    def run(self, _):
        self.clf.fit(self.data)
        self.anomaly_scores = self.clf.decision_scores_[-len(self.X_test):]
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

        del self.clf  # Free memory for the model
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

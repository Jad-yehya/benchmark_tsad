from benchopt import BaseSolver

from importlib.util import find_spec

import numpy as np
import torch
from TSB_AD.model_wrapper import run_TimesFM

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "TSB-TimesFM"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad", "pip::timesfm"]

    parameters = {
        "win_size": [256],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        _, n_features, _ = X_train.shape
        self.data = np.append(X_train, X_test, axis=2)
        self.data = self.data.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

    def skip(self, X_train, X_test):
        if find_spec("timesfm") is None:
            return True, "TSB-TimesFM requires the optional timesfm package."
        return False, None

    def run(self, _):
        anomaly_scores = run_TimesFM(
            data=self.data,
            win_size=self.win_size,
        )
        self.anomaly_scores = anomaly_scores[-len(self.X_test):]
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

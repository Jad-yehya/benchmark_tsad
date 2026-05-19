from benchopt import BaseSolver

import pandas as pd
from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig
from merlion.utils.time_series import TimeSeries

from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "DAGMM"

    install_cmd = "conda"
    requirements = ["pip::salesforce-merlion", "pip::scikit-learn"]

    parameters = {
        "gmm_k": [3],
        "hidden_size": [256],
        "sequence_len": [10],
        "num_epochs": [10],
        "lr": [1e-3],
        "batch_size": [8192],
        "lambda_energy": [0.1],
        "lambda_cov": [0.005],
        "cutoff": [None],
        # "device": ["cuda:3"]
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        n_features = X_train.shape[1]
        self.X_train = X_train.transpose(0, 2, 1).reshape(-1, n_features)
        self.X_test = X_test.transpose(0, 2, 1).reshape(-1, n_features)
        # Convert to Merlion TimeSeries
        # We use a default index since we don't have timestamps
        train_df = pd.DataFrame(self.X_train)
        test_df = pd.DataFrame(self.X_test)

        # Merlion expects a time index or it will generate one
        self.train_data = TimeSeries.from_pd(train_df)
        self.test_data = TimeSeries.from_pd(test_df)

        # Configure DAGMM
        config = DAGMMConfig(
            gmm_k=self.gmm_k,
            hidden_size=self.hidden_size,
            sequence_len=self.sequence_len,
            num_epochs=self.num_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            lambda_energy=self.lambda_energy,
            lambda_cov=self.lambda_cov,
            # device=self.device
        )

        self.model = DAGMM(config)

    def run(self, _):
        # Train
        self.model.train(self.train_data)

        # Predict
        # get_anomaly_score returns a TimeSeries of scores
        scores_ts = self.model.get_anomaly_score(self.test_data)
        self.anomaly_scores = scores_ts.to_pd().values.flatten()
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

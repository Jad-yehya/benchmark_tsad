from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig
    from merlion.utils.time_series import TimeSeries
    from sklearn.preprocessing import MinMaxScaler


class Solver(BaseSolver):
    name = "DAGMM"

    install_cmd = "conda"
    requirements = ["pip:salesforce-merlion", "pip:scikit-learn"]

    parameters = {
        "gmm_k": [3],
        "hidden_size": [256],
        "sequence_len": [10],
        "num_epochs": [10],
        "lr": [1e-3],
        "batch_size": [8192],
        "lambda_energy": [0.1],
        "lambda_cov": [0.005],
        # "device": ["cuda:3"]
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        print(X_train.shape, X_test.shape, y_test.shape)
        print(X_train.dtype, X_test.dtype, y_test.dtype)
        print("Nan in X_train:", np.isnan(
            X_train).any(), np.isnan(X_train).sum())
        print("Nan in X_test:", np.isnan(X_test).any(), np.isnan(X_test).sum())
        print("Nan in y_test:", np.isnan(y_test).any(), np.isnan(y_test).sum())

        n_features = X_train.shape[1]
        self.X_train = X_train.transpose(0, 2, 1).reshape(-1, n_features)
        self.X_test = X_test.transpose(0, 2, 1).reshape(-1, n_features)
        self.y_test = y_test.reshape(-1)
        # Convert to Merlion TimeSeries
        # We use a default index since we don't have timestamps
        train_df = pd.DataFrame(self.X_train)
        test_df = pd.DataFrame(self.X_test)

        print("Dataframe OK")

        # Merlion expects a time index or it will generate one
        self.train_data = TimeSeries.from_pd(train_df)
        self.test_data = TimeSeries.from_pd(test_df)

        print("TimeSeries OK")

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
        self.scores = scores_ts.to_pd().values.flatten()

    def get_result(self):
        # Normalize scores to 0-1 range for thresholding
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores_norm = scaler.fit_transform(self.scores.reshape(-1, 1)).ravel()

        # Simple thresholding
        y_hat = np.where(scores_norm > 0.75, 1, 0)

        return dict(
            y_hat=y_hat,
            raw_anomaly_score=self.scores
        )

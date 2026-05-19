from benchopt import BaseSolver
from sklearn.preprocessing import MinMaxScaler

from benchmark_utils.predictions import cutoff_scores
from benchmark_utils.windowing import find_period_length
from TSB_AD.models.MatrixProfile import MatrixProfile


class Solver(BaseSolver):
    name = "MP"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad", "scikit-learn"]

    parameters = {
        "window_size": [128, "auto"],
        "cutoff": [None],
    }

    test_config = {
        "dataset": {
            "n_features": 1,
        },
        "window_size": 8,
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        # Shapes received: (n_recordings, n_features, n_samples)
        self.X_train = X_train
        self.X_test = X_test

        n_features = X_train.shape[1]

        self.X_train = self.X_train.reshape(-1, n_features)
        self.X_test = self.X_test.reshape(-1, n_features)

        if self.window_size == "auto":
            self.window_size = int(find_period_length(X_train.reshape(-1)))

        self.clf = MatrixProfile(
            window=self.window_size,
        )

    def run(self, _):
        # Special solver, fitting on X_test
        self.clf.fit(self.X_test.reshape(-1))
        anomaly_scores = self.clf.decision_scores_
        self.anomaly_scores = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(anomaly_scores.reshape(-1, 1))
            .ravel()
        )
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def skip(self, X_train, X_test):
        """Check if the solver can be skipped."""
        if (find_period_length(X_train.reshape(-1)) == 0) and (
                self.window_size == "auto"):
            return True, "Window size is 0"
        if X_train.shape[1] != 1:
            return True, "Matrix Profile only supports univariate data"
        return False, None

    def get_result(self):
        """Return the result of the solver."""
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

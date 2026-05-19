from benchopt import BaseSolver

from sklearn.preprocessing import MinMaxScaler

from benchmark_utils.models import Autoencoder
from benchmark_utils.predictions import cutoff_scores
from benchmark_utils.windowing import find_period_length


class Solver(BaseSolver):
    name = "AE"

    install_cmd = "conda"
    requirements = ["pytorch", "scikit-learn", "tqdm"]

    parameters = {
        "window_size": [10, "auto"],
        "num_epochs": [100],
        "batch_size": [1024],
        "learning_rate": [1e-3],
        "hidden_size": [64],
        "latent_size": [32],
        "cutoff": [None],
    }

    test_config = {
        "window_size": 10,
        "num_epochs": 1,
        "batch_size": 8,
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        if self.window_size == "auto":
            self.window_size = find_period_length(X_train.reshape(-1))

        # Data received has shape (n_recordings, n_features, n_samples)
        n_features = X_train.shape[1]
        self.X_train = X_train.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

        # For multivariate data, input_size = window_size * n_features
        self.clf = Autoencoder(
            input_size=self.window_size * n_features,
            sliding_window=self.window_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
        )

    def run(self, _):
        self.clf.fit(
            self.X_train,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=float(self.learning_rate),
        )

        self.clf.predict(self.X_test)
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
        if find_period_length(X_train.reshape(-1)) == 0 and (
            self.window_size == "auto"
        ):
            return True, "Window size is 0"
        return False, None

    def get_result(self):
        """Return the result of the solver."""
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

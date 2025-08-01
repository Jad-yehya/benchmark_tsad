from benchopt import BaseSolver, safe_import_context
from sklearn.preprocessing import MinMaxScaler

with safe_import_context() as import_ctx:
    from benchmark_utils.models import Autoencoder
    from TSB_UAD.utils.slidingWindows import find_length
    import numpy as np


class Solver(BaseSolver):
    name = "AE"

    install_cmd = "conda"
    requirements = ["pip:tsb-uad"]

    parameters = {
        "window_size": [10, "auto"],
        "num_epochs": [100],
        "batch_size": [128],
        "learning_rate": [1e-3],
        "hidden_size": [64],
        "latent_size": [32],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        if self.window_size == "auto":
            self.window_size = find_length(X_train)

        self.X_train = X_train.reshape(-1)
        self.X_test = X_test.reshape(-1)
        self.y_test = y_test

        self.clf = Autoencoder(
            input_size=self.window_size,
            sliding_window=self.window_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
        )

    def run(self, _):
        self.clf.fit(
            self.X_train,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        self.clf.predict(self.X_test.reshape(-1, 1))
        score = self.clf.decision_scores_

        self.score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(score.reshape(-1, 1))
            .ravel()
        )

    def skip(self, X_train, y_test, X_test):
        """Check if the solver can be skipped."""
        if find_length(X_train) == 0 and self.window_size == "auto":
            return True, "Window size is 0"
        return False, None

    def get_result(self):
        """Return the result of the solver."""
        # Binarizing the scores to 0 and 1
        # TEMPORARY SOLUTION
        self.final_score = np.where(self.score > 0.75, 1, 0)
        return dict(y_hat=self.final_score)

from benchopt import BaseSolver
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from TSB_AD.models.MatrixProfile import MatrixProfile
from TSB_AD.utils.slidingWindows import find_length


class Solver(BaseSolver):
    name = "MP"

    install_cmd = "conda"
    requirements = ["pip::tsb-ad", "scikit-learn"]

    parameters = {
        "window_size": [128, "auto"],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        # Shapes received: (n_recordings, n_features, n_samples)
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

        n_features = X_train.shape[1]

        self.X_train = self.X_train.reshape(-1, n_features)
        self.X_test = self.X_test.reshape(-1, n_features)

        if self.window_size == "auto":
            self.window_size = int(find_length(X_train.reshape(-1)))

        print("=====================")
        print(f"window_size: {self.window_size}")
        print("=====================")

        self.clf = MatrixProfile(
            window=self.window_size,
        )

    def run(self, _):
        print("Running Matrix Profile solver...")
        # Special solver, fitting on X_test
        self.clf.fit(self.X_test.reshape(-1))
        print("MP Fitted")
        self.scores = self.clf.decision_scores_
        self.score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.scores.reshape(-1, 1))
            .ravel()
        )
        print("MP Scored")
        print(f"Score shape: {self.score.shape}")

    def skip(self, X_train, y_test, X_test):
        """Check if the solver can be skipped."""
        if (find_length(X_train.reshape(-1)) == 0) and (
                self.window_size == "auto"):
            return True, "Window size is 0"
        if X_train.shape[1] != 1:
            return True, "Matrix Profile only supports univariate data"
        return False, None

    def get_result(self):
        """Return the result of the solver."""
        # Binarizing the scores to 0 and 1
        # TEMPORARY SOLUTION
        self.final_score = np.where(self.score > 0.90, 1, 0)
        return dict(y_hat=self.final_score, raw_anomaly_score=self.score)

from benchopt import BaseSolver, safe_import_context
from sklearn.preprocessing import MinMaxScaler

with safe_import_context() as import_ctx:
    from TSB_UAD.models.matrix_profile import MatrixProfile
    from TSB_UAD.utils.slidingWindows import find_length
    import math
    import numpy as np


class Solver(BaseSolver):
    name = "MP"

    install_cmd = "conda"
    requirements = ["pip:tsb-uad"]

    parameters = {
        "window_size": [10, "auto"],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

        if self.window_size == "auto":
            self.window_size = find_length(X_train)
        self.clf = MatrixProfile(
            window=self.window_size,
        )

    def run(self, _):
        print("Running Matrix Profile solver...")
        # Special solver, fitting on X_test
        self.clf.fit(self.X_test.reshape(-1))
        self.scores = self.clf.decision_scores_
        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.scores.reshape(-1, 1))
            .ravel()
        )

        print("MP Fitted")

        self.score = np.array(
            [score[0]] * math.ceil((self.window_size - 1) / 2)
            + list(score)
            + [score[-1]] * ((self.window_size - 1) // 2)
        )

        print("MP Scored")
        print(f"Score shape: {score.shape}")

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

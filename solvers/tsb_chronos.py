from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from TSB_AD.models.Chronos import Chronos
    from TSB_AD.utils.slidingWindows import find_length
    import numpy as np
    import torch


class Solver(BaseSolver):
    name = "TSB-Chronos"

    install_cmd = "conda"
    requirements = ["pip:tsb-ad"]

    parameters = {
        "win_size": ["auto"],
        "prediction_length": [1],
        "model_size": ['base'],
        "batch_size": [32],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
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
        print("Running Chronos solver...")
        self.clf.fit(self.data)
        self.score = self.clf.decision_scores_[-len(self.X_test):]
        print("Chronos Fitted")

        # Map scores to predictions
        threshold = np.percentile(self.score, (1 - 0.1) * 100)
        self.y_hat = (self.score > threshold).astype(int)
        del self.clf  # Free memory for the model
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        return dict(y_hat=self.y_hat, raw_anomaly_score=self.score)

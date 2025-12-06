from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from TSB_AD.models.TimesNet import TimesNet
    import torch


class Solver(BaseSolver):
    name = "TSB-TimesNet"

    install_cmd = "conda"
    requirements = ["pip:tsb-ad"]

    parameters = {
        "window_size": [256],
        "lr": [1e-4],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        _, n_features, _ = X_train.shape
        self.X_train = X_train.reshape(-1, n_features)
        self.X_test = X_test.reshape(-1, n_features)

        self.clf = TimesNet(
            win_size=self.window_size,
            enc_in=n_features,
            epochs=10,
            batch_size=128,
            lr=self.lr,
            patience=3,
            features="M",
            lradj="type1",
            validation_size=0.2,
        )

    def run(self, _):
        self.clf.fit(self.X_train)
        self.raw_anomaly_score = self.clf.decision_function(self.X_test)

        print("TimesNet done")
        del self.clf.model
        del self.clf
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        self.y_hat = (self.raw_anomaly_score > 0).astype(int)
        return dict(y_hat=self.y_hat, raw_anomaly_score=self.raw_anomaly_score)

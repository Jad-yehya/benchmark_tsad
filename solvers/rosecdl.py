from re import X
from benchopt import safe_import_context, BaseSolver

with safe_import_context() as import_ctx:
    from rosecdl.rosecdl import RoseCDL
    import torch


class Solver(BaseSolver):
    name = "RoseCDL"

    install_cmd = "conda"
    requirements = ["pip:rosecdl", "pip:torch"]

    parameters = {
        "n_components": [1],
        "kernel_size": [64],
        "lmbd": [0.8],
        "scale_lmbd": [False],
        "epochs": [50],
        "max_batch": [None],
        "mini_batch_size": [600],
        "sample_window": [1_000],
        "optimizer": ["adam"],
        "n_iterations": [90],
        "window": [False],
        "outliers_kwargs": [
            {
                "method": "mad",
                "alpha": 3.5,
                "moving_average": None,
                "union_channels": True,
                "opening_window": True,
            },
        ],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # We receive data in shape (n_recordings, n_features, n_samples)
        self.y_test = y_test
        self.X_train = torch.tensor(
            X_train, dtype=torch.float32, device=self.device)
        self.X_test = X_test

        self.clf = RoseCDL(
            n_components=self.n_components,
            n_channels=X_train.shape[1],
            kernel_size=self.kernel_size,
            lmbd=self.lmbd,
            scale_lmbd=self.scale_lmbd,
            epochs=self.epochs,
            max_batch=self.max_batch,
            mini_batch_size=self.mini_batch_size,
            sample_window=self.sample_window,
            optimizer=self.optimizer,
            n_iterations=self.n_iterations,
            window=self.window,
            device=self.device,
            outliers_kwargs=self.outliers_kwargs,
        )

    def run(self, _):
        self.clf.fit(self.X_train)
        self.y_pred = self.clf.get_outlier_mask(self.X_test)

        xh, zh = self.clf.csc(
            torch.tensor(self.X_test, dtype=torch.float32, device=self.device)
        )
        err = self.clf.loss_fn.compute_patch_error(
            X_hat=xh,
            z_hat=zh,
            X=torch.tensor(self.X_test, dtype=torch.float32,
                           device=self.device),
        )
        err = err.cpu().detach().numpy()
        # Aggregate errors over channels
        self.err = err.sum(axis=1).reshape(-1)

    def get_result(self):
        return dict(y_hat=self.y_pred, raw_anomaly_score=self.err)

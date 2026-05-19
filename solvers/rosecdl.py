from benchopt import BaseSolver

import torch
from benchmark_utils.predictions import cutoff_scores
from benchmark_utils.windowing import find_period_length
from rosecdl.rosecdl import RoseCDL


class Solver(BaseSolver):
    name = "RoseCDL"

    install_cmd = "conda"
    requirements = [
        "pytorch", "pip::git+https://github.com/tommoral/rosecdl.git"
    ]

    parameters = {
        "n_components": [1],
        "kernel_size": ["auto"],
        "lmbd": [0.8],
        "scale_lmbd": [False],
        "epochs": [70],
        "max_batch": [None],
        "mini_batch_size": [600],
        "sample_window": [1_000],
        "optimizer": ["linesearch"],
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
        "plot": [False],
        "cutoff": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, X_test):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # We receive data in shape (n_recordings, n_features, n_samples)
        self.X_train = torch.tensor(
            X_train, dtype=torch.float32, device=self.device)
        self.X_test = X_test

        if self.kernel_size == "auto":
            self.kernel_size = int(find_period_length(X_train.reshape(-1)))

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
        del self.X_train  # Free GPU memory for X_train after fitting

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
        self.anomaly_scores = err.sum(axis=1).reshape(-1)
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )
        del self.clf  # Free GPU memory for the model
        torch.cuda.empty_cache()  # Release cached GPU memory

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

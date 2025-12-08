from benchopt import BaseSolver

import torch
import numpy as np
from pyod.models.vae import VAE

from benchmark_utils.windowing import make_windows


class Solver(BaseSolver):
    name = "VAE"

    install_cmd = "conda"
    requirements = ["pip::pyod", "pip::torch"]

    sampling_strategy = "run_once"

    parameters = {
        "contamination": [0.005, 0.05, 0.1, 0.2],
        "n_epochs": [50],
        "window_size": [256],
        "horizon": [0],
        "stride": [1],
        "batch_size": [128],
        "preprocessing": [True],
        "latent_dim": [2, 5, 10],
        "batch_norm": [True],
        "dropout_rate": [0.1, 0.2, 0.5],
    }
    test_config = {
        'solver': {
            "n_epochs": 1,
            "window_size": 16,
        }
    }

    def set_objective(self, X_train, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.X_train = X_train
        self.X_test = X_test

        self.Xw_train = make_windows(
            X_train,
            window_size=self.window_size,
            stride=self.stride
        ).reshape(-1, self.window_size * X_train.shape[1])

        self.Xw_test = make_windows(
            X_test,
            window_size=self.window_size+self.horizon,
            stride=self.stride,
            padding=True
        ).reshape(-1, self.window_size * X_train.shape[1])

        self.clf = VAE(
            contamination=self.contamination,
            preprocessing=self.preprocessing,
            batch_size=min(self.batch_size, len(self.Xw_train)),
            epoch_num=self.n_epochs,
            device=self.device,
            latent_dim=self.latent_dim,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            lr=1e-5
        )

    def run(self, _):
        self.clf.fit(self.Xw_train)
        self.y_pred = self.clf.predict(self.Xw_test)

    def get_result(self):
        return dict(y_hat=self.y_pred)

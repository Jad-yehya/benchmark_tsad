from benchopt import BaseSolver

import torch
import numpy as np
from pyod.models.vae import VAE


class Solver(BaseSolver):
    name = "VAE"

    install_cmd = "conda"
    requirements = ["pip::pyod", "pip::torch"]

    sampling_strategy = "run_once"

    parameters = {
        "contamination": [0.005, 0.05, 0.1, 0.2],
        "n_epochs": [50],
        "window": [False],
        "window_size": [256],
        "horizon": [0],
        "stride": [1],
        "batch_size": [128],
        "preprocessing": [True, False],
        "latent_dim": [2, 5, 10],
        "batch_norm": [True, False],
        "dropout_rate": [0.1, 0.2, 0.5],
    }

    def set_objective(self, X_train, y_test, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

        self.clf = VAE(contamination=self.contamination,
                       preprocessing=self.preprocessing,
                       batch_size=self.batch_size,
                       epoch_num=self.n_epochs,
                       device=self.device,
                       latent_dim=self.latent_dim,
                       batch_norm=self.batch_norm,
                       dropout_rate=self.dropout_rate,
                       )

        if self.window:
            self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                X_train,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

            if self.X_test is not None:
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    X_test,
                    window_shape=self.window_size+self.horizon,
                    axis=0
                ).transpose(0, 2, 1)

            if self.y_test is not None:
                self.yw_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=0
                )[::self.stride]

                self.yw_test = torch.tensor(
                    self.yw_test, dtype=torch.float32
                )
        else:
            self.Xw_train = X_train
            self.Xw_test = X_test

    def run(self, _):
        self.clf.fit(self.Xw_train)
        self.y_pred = self.clf.predict(self.Xw_test)

    def get_result(self):
        return dict(y_hat=self.y_pred)

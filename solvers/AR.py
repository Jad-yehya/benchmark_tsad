# AR model
from benchopt import BaseSolver

import torch
from torch import optim, nn
import numpy as np
from tqdm import tqdm

from benchmark_utils.models import ARModel
from benchmark_utils import mean_overlaping_pred
from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "AR"  # AutoRegressive Linear model

    install_cmd = "conda"
    requirements = ["pytorch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "batch_size": [128],
        "n_epochs": [50],
        "lr": [1e-5],
        "weight_decay": [1e-7],
        "window_size": [100],
        "horizon": [1],
        "cutoff": [None],
    }

    test_config = {
        "n_epochs": 1,
        "window_size": 16,
    }

    def set_objective(self, X_train, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Receiving shapes of (n_recordings, n_features, n_samples)

        _, n_features, _ = X_train.shape

        # (n_samples, n_features)
        self.X_train = X_train.reshape(-1, n_features)
        # (n_samples, n_features)
        self.X_test = X_test.reshape(-1, n_features)

        self.model = ARModel(
            n_features,
            self.window_size,
            self.horizon
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
            # weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()

        if self.X_train is not None:
            # (n_windows, window_size+horizon, n_features)
            self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                self.X_train,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

        if self.X_test is not None:
            # (n_windows, window_size+horizon, n_features)
            self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                self.X_test,
                window_shape=self.window_size+self.horizon,
                axis=0
            ).transpose(0, 2, 1)

    def run(self, _):

        self.model.to(self.device)
        self.criterion.to(self.device)

        best_loss = float('inf')  # Initialize best_loss with infinity
        best_model = None         # Variable to store the best model

        ti = tqdm(range(self.n_epochs), desc="epoch", leave=True)

        for epoch in ti:
            self.model.train()
            epoch_loss = 0.0

            for i in range(0, len(self.Xw_train), self.batch_size):
                # (batch_size, window_size, n_features)
                x = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.window_size, :],
                    dtype=torch.float32, device=self.device
                )
                # (batch_size, horizon, n_features)
                y = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.horizon, :],
                    dtype=torch.float32, device=self.device
                )

                self.optimizer.zero_grad()
                y_pred = self.model(x)  # (batch_size, horizon, n_features)
                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(self.Xw_train) / self.batch_size)
            ti.set_description(f"Epoch {epoch}, Epoch Loss {epoch_loss: .5e}")

            # Checkpoint the model if the loss is lower
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = self.model.state_dict()

        self.model.load_state_dict(best_model)

        self.model.eval()
        # (n_test_windows, horizon, n_features)
        xw_hat = self.model(torch.tensor(
            self.Xw_test[:, :self.window_size, :],
            dtype=torch.float32
        ).to(self.device))

        xw_hat = xw_hat.detach().cpu().numpy()

        # Reconstructing the prediction from the predicted windows.
        # The first ``window_size`` positions have no forecast (no full input
        # window precedes them); fill them with -1 as a sentinel.
        x_hat = np.zeros_like(self.X_test) - 1
        x_hat[self.window_size:] = mean_overlaping_pred(xw_hat, 1)

        reconstruction_err = np.abs(
            self.X_test[self.window_size:] - x_hat[self.window_size:]
        )
        self.anomaly_scores = np.full(
            self.X_test.shape, np.nan, dtype=float
        )
        self.anomaly_scores[self.window_size:] = reconstruction_err
        self.anomaly_scores = np.max(self.anomaly_scores, axis=1)

        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    # Skipping the solver call if a condition is met
    def skip(self, X_train, X_test):
        if X_train.shape[0]*X_train.shape[2] < self.window_size + self.horizon:
            return True, "No enough training samples"
        if X_test.shape[0]*X_test.shape[2] < self.window_size + self.horizon:
            return True, "No enough testing samples"
        return False, None

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

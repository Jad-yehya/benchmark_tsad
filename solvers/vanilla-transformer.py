# Vanilla Transformer
from benchopt import BaseSolver

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from benchmark_utils.models import TransformerModel
from benchmark_utils.windowing import make_windowed_dataset
from benchmark_utils.windowing import reconstruct_from_windows
from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "Transformer"

    install_cmd = "conda"
    requirements = ["pytorch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "num_layers": [1],
        "num_heads": [2],
        "dim_feedforward": [512],
        "batch_size": [32],
        "n_epochs": [50],
        "lr": [1e-5],
        "horizon": [1],
        "window_size": [256],
        "stride": [1],
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

        self.X_train = X_train
        self.X_test = X_test

        self.model = TransformerModel(
            n_features=X_train.shape[1],
            sequence_length=self.window_size,
            horizon=self.horizon,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.Xw_train = make_windowed_dataset(
            X_train,
            window_size=self.window_size+self.horizon,
            stride=self.stride
        )
        self.Xw_test = make_windowed_dataset(
            X_test,
            window_size=self.window_size+self.horizon,
            stride=self.stride
        )
        self.train_loader = DataLoader(
            self.Xw_train, batch_size=self.batch_size, shuffle=True,
        )
        self.test_loader = DataLoader(
            self.Xw_test, batch_size=self.batch_size, shuffle=False,
        )

    def run(self, _):
        self.model.to(self.device)
        self.model.train()

        ti = tqdm(range(self.n_epochs), desc="epoch", leave=True)

        best_loss = np.inf
        patience = 20
        no_improve = 0

        # Training loop
        for epoch in ti:
            self.model.train()
            total_loss = 0
            for x, in self.train_loader:
                x = x.to(self.device)
                y = x[:, -self.horizon:]
                x = x[:, :-self.horizon]

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                total_loss += loss.item()

                avg_loss = total_loss / (len(self.Xw_train) // self.batch_size)
                ti.set_description(
                    f"Epoch {epoch} (loss={avg_loss:.5e})"
                )

                # Learning rate scheduling
                self.scheduler.step(avg_loss)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve == patience:
                        break

        # Test loop
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for x, in self.test_loader:
                batch = x[:, :self.window_size].to(self.device)
                with torch.no_grad():
                    batch_predictions = self.model(batch)
                all_predictions.append(batch_predictions.cpu().numpy())

        xw_hat = np.concatenate(all_predictions, axis=0)

        # Continue with the rest of your code for reconstructing predictions
        x_hat = np.zeros_like(self.X_test) - 1
        x_hat[..., self.window_size:] = reconstruct_from_windows(
            xw_hat, stride=self.stride, batch=len(self.X_test),
            n_features=self.X_test.shape[1]
        )

        reconstruction_err = np.abs(
            self.X_test[..., self.window_size:] - x_hat[..., self.window_size:]
        )
        self.anomaly_scores = np.full(
            self.X_test.shape[:1] + self.X_test.shape[2:],
            np.nan,
            dtype=float,
        )
        self.anomaly_scores[..., self.window_size:] = np.max(
            reconstruction_err, axis=1
        )

        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def skip(self, X_train, X_test):
        if X_train.shape[-1] < self.window_size + self.horizon:
            return True, "No enough training samples"
        return False, None

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

# Vanilla Transformer
from benchopt import BaseSolver

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from benchmark_utils import mean_overlaping_pred
from benchmark_utils.models import TransformerModel


class Solver(BaseSolver):
    name = "Transformer"

    install_cmd = "conda"
    requirements = ["pip::torch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "num_layers": [1],
        "num_heads": [2],
        "dim_feedforward": [512],
        "batch_size": [32],
        "n_epochs": [50],
        "lr": [1e-5],
        "horizon": [1],
        "window": [True],
        "window_size": [256],
        "stride": [1],
        "percentile": [97],
    }

    def set_objective(self, X_train, y_test, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test

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

        # Using only windowed data, parameter used only for consistency
        if self.window:
            if self.X_train is not None:
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
            for i in range(0, len(self.Xw_train), self.batch_size):
                x = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, :self.window_size, :],
                    dtype=torch.float32).to(self.device)
                y = torch.tensor(
                    self.Xw_train[i:i+self.batch_size, -self.horizon:, :],
                    dtype=torch.float32).to(self.device)

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
                ti.set_description(f"Epoch {epoch} (loss={avg_loss:.5e})")

                # Learning rate scheduling
                self.scheduler.step(avg_loss)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    no_improve += 1
                    if no_improve == patience:
                        break

        # Test loop
        self.model.eval()
        batch_size = 1024
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(self.Xw_test), batch_size):
                batch = torch.tensor(
                    self.Xw_test[i:i+batch_size, :self.window_size, :],
                    dtype=torch.float32
                ).to(self.device)

                batch_predictions = self.model(batch)

                if batch_predictions.is_cuda:
                    batch_predictions = batch_predictions.cpu().numpy()
                else:
                    batch_predictions = batch_predictions.numpy()

                all_predictions.append(batch_predictions)

        xw_hat = np.concatenate(all_predictions, axis=0)

        # Continue with the rest of your code for reconstructing predictions
        x_hat = np.zeros_like(self.X_test) - 1
        x_hat[self.window_size:self.window_size+self.horizon] = xw_hat[0]
        x_hat[self.window_size+self.horizon:] = mean_overlaping_pred(
            xw_hat, 1)

        # Calculating the percentile value for the threshold
        percentile_value = np.percentile(
            np.abs(self.X_test[self.window_size:] - x_hat[self.window_size:]),
            self.percentile
        )

        # Thresholding
        predictions = np.zeros_like(x_hat)-1
        predictions[self.window_size:] = np.where(
            np.abs(self.X_test[self.window_size:] -
                   x_hat[self.window_size:]) > percentile_value, 1, 0
        )

        self.predictions = np.max(predictions, axis=1)

    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0] < self.window_size + self.horizon:
            return True, "No enough training samples"
        return False, None

    def get_result(self):
        return dict(y_hat=self.predictions)

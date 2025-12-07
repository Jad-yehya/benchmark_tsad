# LSTM Autoencoder
from benchopt import BaseSolver

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from benchmark_utils.models import AutoEncoderLSTM


class Solver(BaseSolver):
    name = "LSTM"

    install_cmd = "conda"
    requirements = ["pip::torch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "embedding_dim": [64],
        "batch_size": [32],
        "n_epochs": [50],
        "lr": [1e-5],
        "window": [True],
        "window_size": [256],  # window_size = seq_len
        "stride": [1],
        "percentile": [97],
        "encoder_layers": [32],
        "decoder_layers": [32],
    }

    def prepare_data(self, *data):
        # return tensors on device
        return (torch.tensor(
            d, dtype=torch.float32, device=self.device)
            for d in data)

    def set_objective(self, X_train, y_test, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.X_train = X_train
        self.X_test, self.y_test = X_test, y_test
        self.n_features = X_train.shape[1]
        self.seq_len = self.window_size

        self.model = AutoEncoderLSTM(
            n_features=self.n_features,
            sequence_length=self.seq_len,
            embedding_dim=self.embedding_dim,
            enc_layers=self.encoder_layers,
            dec_layers=self.decoder_layers,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        if self.window:
            if self.X_train is not None:
                self.Xw_train = np.lib.stride_tricks.sliding_window_view(
                    self.X_train, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

                self.Xw_train = torch.tensor(
                    self.Xw_train, dtype=torch.float32
                )

            if self.X_test is not None:
                self.Xw_test = np.lib.stride_tricks.sliding_window_view(
                    self.X_test, window_shape=self.window_size, axis=0
                )[::self.stride].transpose(0, 2, 1)

                self.Xw_test = torch.tensor(
                    self.Xw_test, dtype=torch.float32
                )

            if self.y_test is not None:
                self.yw_test = np.lib.stride_tricks.sliding_window_view(
                    self.y_test, window_shape=self.window_size, axis=0
                )[::self.stride]

                self.yw_test = torch.tensor(
                    self.yw_test, dtype=torch.float32
                )

        self.train_loader = DataLoader(
            self.Xw_train, batch_size=self.batch_size, shuffle=True,
        )
        self.test_loader = DataLoader(
            self.Xw_test, batch_size=self.batch_size, shuffle=False,
        )

    def run(self, _):

        self.model.to(self.device)
        self.criterion.to(self.device)

        ti = tqdm(range(self.n_epochs), desc="epoch", leave=True)

        # Training loop
        for epoch in ti:
            self.model.train()
            train_loss = 0
            for i, x in enumerate(self.train_loader):

                x = x.to(self.device)

                self.optimizer.zero_grad()
                x_hat = self.model(x)

                loss = self.criterion(x, x_hat)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            ti.set_postfix(train_loss=f"{train_loss:.5f}")

        # Saving the model
        torch.save(self.model.state_dict(), "model.pth")

        # Test loop
        self.model.eval()
        raw_reconstruction = []
        for x in self.test_loader:

            x = x.to(self.device)

            x_hat = self.model(x)
            raw_reconstruction.append(x_hat.detach().cpu().numpy())

        raw_reconstruction = np.concatenate(raw_reconstruction, axis=0)

        reconstructed_data = np.concatenate(
            [raw_reconstruction[0], raw_reconstruction[1:, -1, :]], axis=0
        )

        reconstruction_err = np.mean(
            np.abs(self.X_test - reconstructed_data), axis=1
        )

        self.y_hat = np.where(
            reconstruction_err > np.percentile(
                reconstruction_err, self.percentile), 1, 0
        )

    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0] < self.window_size:
            return True, "Not enough samples to create a window."
        return False, None

    def get_result(self):
        return dict(y_hat=self.y_hat)

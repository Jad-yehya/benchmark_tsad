# LSTM Autoencoder
from benchopt import BaseSolver

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from benchmark_utils.models import AutoEncoderLSTM
from benchmark_utils.windowing import make_windowed_dataset
from benchmark_utils.windowing import reconstruct_from_windows
from benchmark_utils.predictions import cutoff_scores


class Solver(BaseSolver):
    name = "LSTM"

    install_cmd = "conda"
    requirements = ["pytorch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "embedding_dim": [64],
        "batch_size": [32],
        "n_epochs": [50],
        "lr": [1e-5],
        "window_size": [256],  # window_size = seq_len
        "stride": [1],
        "cutoff": [None],
        "encoder_layers": [32],
        "decoder_layers": [32],
    }

    test_config = {
        "embedding_dim": 2,
        "batch_size": 1,
        "n_epochs": 1,
        "window_size": 16,
    }

    def set_objective(self, X_train, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.X_train = X_train
        self.X_test = X_test
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

        self.Xw_train = make_windowed_dataset(
            self.X_train, window_size=self.window_size,
            stride=self.stride
        )

        self.Xw_test = make_windowed_dataset(
            self.X_test, window_size=self.window_size,
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
        self.criterion.to(self.device)

        ti = tqdm(range(self.n_epochs), desc="epoch", leave=True)

        # Training loop
        for epoch in ti:
            self.model.train()
            train_loss = 0
            for x, in self.train_loader:

                x = x.to(self.device)

                self.optimizer.zero_grad()
                x_hat = self.model(x)

                loss = self.criterion(x, x_hat)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            ti.set_postfix(train_loss=f"{train_loss:.5f}")

        # Test loop
        self.model.eval()
        raw_reconstruction = []
        for x, in self.test_loader:

            x = x.to(self.device)
            with torch.no_grad():
                x_hat = self.model(x)
            raw_reconstruction.append(x_hat.detach().cpu().numpy())
        reconstructed_data = np.concatenate(raw_reconstruction, axis=0)
        reconstructed_data = reconstruct_from_windows(
            reconstructed_data, stride=self.stride,
            batch=len(self.X_test), n_features=self.n_features
        )

        reconstruction_err = np.mean(
            np.abs(self.X_test - reconstructed_data), axis=1
        )
        self.anomaly_scores = reconstruction_err

        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def skip(self, X_train, X_test):
        if X_train.shape[-1] < self.window_size:
            return True, "Not enough samples to create a window."
        return False, None

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result

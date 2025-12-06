# AR model
from benchopt import BaseSolver, safe_import_context
from benchmark_utils import mean_overlaping_pred

with safe_import_context() as import_ctx:
    import torch
    from torch import optim, nn
    import numpy as np
    from tqdm import tqdm
    from benchmark_utils.models import ARModel


class Solver(BaseSolver):
    name = "AR"  # AutoRegressive Linear model

    install_cmd = "conda"
    requirements = ["pip:torch", "tqdm"]

    sampling_strategy = "run_once"

    parameters = {
        "batch_size": [128],
        "n_epochs": [50],
        "lr": [1e-5],
        "weight_decay": [1e-7],
        "window_size": [100],
        "horizon": [1],
        "percentile": [99.4],
    }

    def set_objective(self, X_train, y_test, X_test):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Receiving shapes of (n_recordings, n_features, n_samples)

        _, n_features, _ = X_train.shape

        # (n_samples, n_features)
        self.X_train = X_train.reshape(-1, n_features)
        # (n_samples, n_features)
        self.X_test = X_test.reshape(-1, n_features)
        self.y_test = y_test.reshape(-1)                # (n_samples,)

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

        print("IN AR")
        print("X_train shape", self.X_train.shape)
        print("X_test shape", self.X_test.shape)
        print("y_test shape", self.y_test.shape)

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

        # Reconstructing the prediction from the predicted windows
        # Creating the prediction array with -1 for the unknown values
        # Corresponding to the first window_size values
        x_hat = np.zeros_like(self.X_test)-1  # (n_test_samples, n_features)
        x_hat[self.window_size:self.window_size+self.horizon] = xw_hat[0]

        x_hat[self.window_size+self.horizon:] = mean_overlaping_pred(
            xw_hat, 1
        )

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

    # Skipping the solver call if a condition is met
    def skip(self, X_train, X_test, y_test):
        if X_train.shape[0]*X_train.shape[2] < self.window_size + self.horizon:
            return True, "No enough training samples"
        if X_test.shape[0]*X_test.shape[2] < self.window_size + self.horizon:
            return True, "No enough testing samples"
        return False, None

    def get_result(self):
        return dict(y_hat=self.predictions)

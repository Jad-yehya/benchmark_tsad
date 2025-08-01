from torch import nn
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


class ARModel(nn.Module):
    """
    Class for the AR Solver
    Single linear layer for autoregressive model
    Taking in input a window of size window_size and
    outputting a window of size horizon
    input : (batch_size, window_size, n_features)
    output : (batch_size, horizon, n_features)
    """

    def __init__(self,
                 n_features: int,
                 window_size: int,
                 horizon=1,
                 ):
        super(ARModel, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.horizon = horizon
        self.linear = nn.Linear(window_size * n_features, horizon * n_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(x.size(0), -1, self.n_features)
        return x


class TransformerModel(nn.Module):
    """
    Class for the Vanilla-Transformer Solver
    Transformer model for time series forecasting
    input : (batch_size, sequence_length, n_features)
    output : (batch_size, horizon, n_features)
    """

    def __init__(self,
                 n_features: int,
                 sequence_length: int,
                 horizon=1,
                 num_layers=1,
                 num_heads=2,
                 dim_feedforward=512,
                 ):
        super(TransformerModel, self).__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.horizon = horizon

        # Ensure d_model is divisible by num_heads
        self.d_model = ((n_features - 1) // num_heads + 1) * num_heads

        self.input_projection = nn.Linear(n_features, self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(
            self.d_model * sequence_length, horizon * n_features)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, n_features)
        src = self.input_projection(src)  # Project to d_model
        src = src.transpose(0, 1)  # (sequence_length, batch_size, d_model)

        output = self.transformer_encoder(src)
        # (batch_size, sequence_length, d_model)
        output = output.transpose(0, 1)
        output = output.flatten(1)  # (batch_size, sequence_length * d_model)
        output = self.fc_out(output)
        # (batch_size, horizon, n_features)
        output = output.view(-1, self.horizon, self.n_features)

        return output


class AutoEncoderLSTM(nn.Module):
    """
    Class for the LSTM Solver
    LSTM Autoencoder model for time series forecasting
    input : (batch_size, sequence_length, n_features)
    output : (batch_size, sequence_length, n_features)
    """

    def __init__(self,
                 n_features: int,
                 sequence_length: int,
                 embedding_dim=64,
                 enc_layers=1,
                 dec_layers=1,
                 ):
        super(AutoEncoderLSTM, self).__init__()
        self.sequence_length, self.n_features = sequence_length, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=enc_layers,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=n_features,
            num_layers=dec_layers,
            batch_first=True
        )

    def forward(self, x):

        x, (_, _) = self.encoder(x)
        x, (_, _) = self.decoder(x)

        return x


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return window  # Input and target are the same for autoencoder


class Autoencoder(nn.Module):
    def __init__(
            self,
            input_size=32,
            hidden_size=32,
            latent_size=16,
            sliding_window=10
    ):
        super(Autoencoder, self).__init__()

        self.sliding_window = sliding_window
        self.decision_scores_ = None

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(latent_size),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)

        # Encode
        encoded = self.encoder(x)

        # Decode
        decoded = self.decoder(encoded)

        return decoded

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def _create_sliding_windows(self, X):
        """Create sliding windows from input data"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # If X is 1D, reshape to 2D
        if X.dim() == 1:
            X = X.unsqueeze(1)

        windows = []
        for i in range(len(X) - self.sliding_window + 1):
            window = X[i:i + self.sliding_window].flatten()
            windows.append(window)

        return torch.stack(windows)

    def fit(
        self,
        X,
        num_epochs=50,
        learning_rate=1e-3,
        device="cuda",
        batch_size=32
    ):
        """
        Train the autoencoder on the provided data.

        Args:
            X: Input data tensor or numpy array shape (n_samples, n_features)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
            batch_size: Batch size for training

        Returns:
            List of training losses per epoch
        """
        # Convert to tensor if numpy array
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # Ensure X is 2D
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.dim() == 3:
            # (n_samples, n_timesteps, n_features)
            X = X.view(-1, 1)

        # Create sliding windows
        windowed_data = self._create_sliding_windows(X)

        # Create dataset and dataloader
        # window_size=1 since we already created windows
        dataset = SlidingWindowDataset(windowed_data, window_size=1)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        losses = []

        # Progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            epoch_loss = 0.0

            # Progress bar for batches
            batch_pbar = tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for batch_idx, (data) in enumerate(batch_pbar):
                data = data.to(device)

                # Forward pass
                output = self(data)
                loss = criterion(output, data)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Update batch progress bar
                batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            # Update epoch progress bar
            epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        return losses

    def predict(self, X_test, X_dirty=None, device="cuda"):
        """
        Predict anomaly scores for time series data.

        Args:
            X_test: Test data for reconstruction
            X_dirty: Original dirty data (if None, uses X_test)
            device: Device to run inference on

        Returns:
            Reconstructed data and sets decision_scores_ attribute
        """
        self.eval()
        self.to(device)

        # Create sliding windows for test data
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        windowed_test = self._create_sliding_windows(X_test)
        windowed_test = windowed_test.to(device)

        with torch.no_grad():
            test_predict = self(windowed_test).cpu().numpy()

        # Calculate MAE loss
        test_mae_loss = np.mean(
            np.abs(test_predict - windowed_test.cpu().numpy()), axis=1)

        # Normalize MAE loss
        nor_test_mae_loss = MinMaxScaler().fit_transform(
            test_mae_loss.reshape(-1, 1)).ravel()

        # Use X_dirty if provided, otherwise use original X_test
        if X_dirty is None:
            if isinstance(X_test, torch.Tensor):
                X_dirty = X_test.cpu().numpy()
            else:
                X_dirty = X_test

        # Initialize score array
        score = np.zeros(len(X_dirty))

        # Fill the score array with sliding window approach
        score[self.sliding_window // 2:self.sliding_window //
              2 + len(test_mae_loss)] = nor_test_mae_loss
        score[:self.sliding_window // 2] = nor_test_mae_loss[0]
        score[self.sliding_window // 2 +
              len(test_mae_loss):] = nor_test_mae_loss[-1]

        # Store decision scores
        self.decision_scores_ = score

        return test_predict

    def encode_data(self, x, device="cuda"):
        """
        Encode input data to latent representation.

        Args:
            x: Input tensor or numpy array
            device: Device to run inference on

        Returns:
            Encoded data as numpy array
        """
        self.eval()
        self.to(device)

        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(device)
        with torch.no_grad():
            encoded = self.encode(x)
        return encoded.cpu().numpy()

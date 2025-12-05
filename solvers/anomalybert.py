from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import sys
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    # Add AnomalyBERT to path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AnomalyBERT'))

    from models.anomaly_transformer import get_anomaly_transformer

class Solver(BaseSolver):
    name = "AnomalyBERT"
    sampling_strategy = "run_once"

    requirements = ["pip:timm","pip:torch", "pip:numpy", "pip:tqdm"]

    parameters = {
        "patch_size": [1],
        "d_embed": [512],
        "n_layer": [6],
        "batch_size": [128],
        "lr": [0.0001],
        "max_steps": [5000],
        "n_patches": [512],
        "seed": [548920],
        "device": ["cuda:1"],
        "window_sliding": [16],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        # X_train shape: (n_series, n_features, n_samples)
        # We assume single series or concatenate them
        if X_train.ndim == 3:
            # Reshape to (n_samples, n_features)
            # Assuming (1, n_features, n_samples) -> (n_samples, n_features)
            # Or (n_series, n_features, n_samples) -> (n_series * n_samples, n_features)
            # But we need to be careful about time continuity if we concatenate.
            # For now, let's assume standard benchopt format which seems to be (1, n_features, n_samples)
            self.X_train = np.transpose(X_train, (0, 2, 1)).reshape(-1, X_train.shape[1]).astype(np.float32)
            self.X_test = np.transpose(X_test, (0, 2, 1)).reshape(-1, X_test.shape[1]).astype(np.float32)
        else:
            self.X_train = X_train.astype(np.float32)
            self.X_test = X_test.astype(np.float32)

    def run(self, _):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        train_data = self.X_train
        d_data = train_data.shape[1]

        # Configuration
        patch_size = self.patch_size
        n_patches = self.n_patches # This corresponds to n_features (max_seq_len) in AnomalyBERT
        data_seq_len = n_patches * patch_size

        if len(train_data) <= data_seq_len:
             raise ValueError(f"Data length {len(train_data)} is smaller than sequence length {data_seq_len}")        # Model
        self.model = get_anomaly_transformer(
            input_d_data=d_data,
            output_d_data=1, # BCE loss
            patch_size=patch_size,
            d_embed=self.d_embed,
            hidden_dim_rate=4.,
            max_seq_len=n_patches,
            positional_encoding=None,
            relative_position_embedding=True,
            transformer_n_layer=self.n_layer,
            transformer_n_head=8,
            dropout=0.1
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_steps, eta_min=self.lr*0.01)

        train_loss_fn = nn.BCELoss().to(device)
        sigmoid = nn.Sigmoid().to(device)

        # Data Augmentation Parameters
        replacing_rate = (0.015, 0.15)

        replacing_table = list(np.random.randint(int(data_seq_len*replacing_rate[0]), int(data_seq_len*replacing_rate[1]), size=10000))
        replacing_table_index = 0
        replacing_table_length = 10000

        soft_replacing_prob = 1 - 0.5
        uniform_replacing_prob = soft_replacing_prob - 0.15
        peak_noising_prob = uniform_replacing_prob - 0.15

        replacing_weight = 0.7

        def replacing_weights(interval_len):
            warmup_len = interval_len // 10
            return np.concatenate((np.linspace(0, replacing_weight, num=warmup_len),
                                   np.full(interval_len-2*warmup_len, replacing_weight),
                                   np.linspace(replacing_weight, 0, num=warmup_len)), axis=None)

        valid_index_list = np.arange(len(train_data) - data_seq_len)
        numerical_column = np.arange(d_data) # Assume all numerical

        # Training Loop
        self.model.train()
        for i in tqdm(range(self.max_steps)):
            first_index = np.random.choice(valid_index_list, size=self.batch_size)
            x = []
            for j in first_index:
                x.append(torch.Tensor(train_data[j:j+data_seq_len].copy()).to(device))

            # Replace data logic
            current_index = replacing_table_index
            replacing_table_index += self.batch_size

            if replacing_table_index > replacing_table_length:
                replacing_lengths = replacing_table[current_index:] + replacing_table[:replacing_table_index-replacing_table_length]
                replacing_table_index -= replacing_table_length
            else:
                replacing_lengths = replacing_table[current_index:replacing_table_index]
                if replacing_table_index == replacing_table_length:
                    replacing_table_index = 0

            replacing_lengths = np.array(replacing_lengths)
            # replacing_index = np.random.randint(0, (len(train_data)-replacing_lengths+1)[:, np.newaxis], size=(self.batch_size, d_data))
            # Simplified replacing index to just pick random start points in train_data
            # Note: train.py uses replacing_data which defaults to train_data

            target_index = np.random.randint(0, data_seq_len-replacing_lengths+1)

            replacing_type = np.random.uniform(0., 1., size=(self.batch_size,))
            replacing_dim_numerical = np.random.uniform(0., 1., size=(self.batch_size, d_data))
            replacing_dim_numerical = replacing_dim_numerical - np.maximum(replacing_dim_numerical.min(axis=1, keepdims=True), 0.3) <= 0.001

            x_anomaly = torch.zeros(self.batch_size, data_seq_len, device=device)

            for j, tar, leng, typ, dim_num in zip(range(self.batch_size), target_index, replacing_lengths, replacing_type, replacing_dim_numerical):
                if leng > 0:
                    _x = x[j].clone().transpose(0, 1) # (d_data, seq_len)

                    # External interval replacing
                    if typ > soft_replacing_prob:
                        col_num = numerical_column[dim_num]
                        if len(col_num) > 0:
                            # Pick random interval from train_data
                            rep_start = np.random.randint(0, len(train_data) - leng)
                            random_interval = train_data[rep_start:rep_start+leng, col_num].copy()

                            # Random flip
                            if np.random.rand() > 0.5: # Horizontal
                                random_interval = random_interval[::-1].copy()
                            if np.random.rand() > 0.5: # Vertical
                                random_interval = 1 - random_interval # Assuming normalized data?

                            _x_temp = torch.from_numpy(random_interval).to(device).transpose(0, 1) # (n_cols, leng)

                            weights = torch.from_numpy(replacing_weights(leng)).float().unsqueeze(0).to(device)
                            _x[col_num, tar:tar+leng] = _x_temp * weights + _x[col_num, tar:tar+leng] * (1 - weights)

                            x_anomaly[j, tar:tar+leng] = 1
                            x[j] = _x.transpose(0, 1)

                    # Uniform replacing
                    elif typ > uniform_replacing_prob:
                        col_num = numerical_column[dim_num]
                        if len(col_num) > 0:
                            _x[col_num, tar:tar+leng] = torch.rand(len(col_num), leng, device=device)
                            x_anomaly[j, tar:tar+leng] = 1
                            x[j] = _x.transpose(0, 1)

                    # Peak noising
                    elif typ > peak_noising_prob:
                        col_num = numerical_column[dim_num]
                        if len(col_num) > 0:
                            peak_index = np.random.randint(0, leng)
                            peak_value = (_x[col_num, tar+peak_index] < 0.5).float().to(device)
                            peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(len(col_num), device=device)
                            _x[col_num, tar+peak_index] = peak_value

                            tar_first = np.maximum(0, tar + peak_index - patch_size)
                            tar_last = tar + peak_index + patch_size + 1
                            x_anomaly[j, tar_first:tar_last] = 1
                            x[j] = _x.transpose(0, 1)

            z = torch.stack(x)
            y = self.model(z)
            y = y.squeeze(-1)
            loss = train_loss_fn(sigmoid(y), x_anomaly)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    def get_result(self):
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.model.eval()

        test_data = self.X_test
        window_size = self.n_patches * self.patch_size
        window_sliding = self.window_sliding # Default from estimate.py
        batch_size = self.batch_size

        # We will just slide over the test data

        n_samples = len(test_data)
        output_values = torch.zeros(n_samples, device=device)
        n_overlap = torch.zeros(n_samples, device=device)

        sigmoid = nn.Sigmoid().to(device)

        with torch.no_grad():
            # Pad test data if needed or just handle boundaries
            # estimate.py handles divisions. We'll assume one continuous sequence.

            # We need to batch the sliding windows
            indices = list(range(0, n_samples - window_size + 1, window_sliding))

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = []
                for idx in batch_indices:
                    x_batch.append(test_data[idx:idx+window_size])

                if not x_batch:
                    continue

                x_batch = torch.Tensor(np.stack(x_batch)).to(device)
                y_batch = sigmoid(self.model(x_batch)).squeeze(-1) # (batch, window_size)

                for j, idx in enumerate(batch_indices):
                    output_values[idx:idx+window_size] += y_batch[j]
                    n_overlap[idx:idx+window_size] += 1

        n_overlap[n_overlap == 0] = 1
        scores = (output_values / n_overlap).cpu().numpy()

        threshold = np.percentile(scores, (1 - 0.1) * 100)
        y_hat = (scores > threshold).astype(int)

        return dict(y_hat=y_hat, raw_anomaly_score=scores)
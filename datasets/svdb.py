from benchopt import BaseDataset

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark_utils.download import fetch_tsb_uad


def load_data(db_path, record_ids=None, verbose=False, number=-1):
    """
    Load data from the database path for specified record IDs.

    Args:
        db_path: Path to the database directory
        record_ids: List of record IDs to load.
        If None, loads all available records.

    Returns:
        tuple: (X, y_true) where:
            - X: numpy array of shape (num_records, num_samples)
            - y_true: numpy array of shape (num_records, num_samples)
    """
    db_path = Path(db_path)

    if record_ids is not None and number > 0:
        print("Warning: 'number' parameter is "
              "ignored when 'record_ids' is provided.")

    if record_ids is None:
        record_files = list(db_path.glob("*.test.csv@*.out"))
        record_ids = [f.name.split(".")[0] for f in record_files]
        if number > 0:
            record_ids = record_ids[:number]

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Handle case where record_id already includes the pattern
        record_files = list(db_path.glob(f"{record_id}*test.csv@*.out"))
        if record_files:
            if len(record_files) > 1:
                if verbose:
                    print(
                        f"Multiple files found for record ID {record_id}, "
                        f"using the first one: {record_files[0]}"
                    )
            record_file = record_files[0]
            # Load the record data
            record_data = pd.read_csv(
                db_path / record_file, header=None).dropna().to_numpy()
            # Assuming first column is the data, second column is labels
            if verbose:
                print(
                    f"Loaded record {record_id} "
                    f"with shape {record_data.shape}")
            if record_data.shape[1] >= 2:
                if verbose:
                    print(f"Record {record_id} has sufficient columns")
                data_list.append(record_data[:, 0].astype(float))
                labels_list.append(record_data[:, 1].astype(int))
            else:
                if verbose:
                    print(f"Insufficient columns for record {record_id}")
        else:
            if verbose:
                print(f"Record file not found for ID: {db_path / record_id}")
    if not data_list:
        raise ValueError("No valid data found")

    max_length = max(len(data) for data in data_list)

    padded_data = []
    padded_labels = []
    for data, labels in zip(data_list, labels_list):
        if len(data) < max_length:
            # Padding with last value for data and 0 for labels
            padded_data.append(
                np.pad(
                    data,
                    (0, max_length - len(data)),
                    mode="constant",
                    constant_values=data[-1],
                )
            )
            padded_labels.append(
                np.pad(
                    labels,
                    (0, max_length - len(labels)),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            padded_data.append(data[:max_length])
            padded_labels.append(labels[:max_length])

    return np.array(padded_data), np.array(padded_labels)


class Dataset(BaseDataset):
    name = "SVDB"

    requirements = ["pip:pooch"]

    parameters = {
        "recordings_id": [["801"]],
        "number": [-1],
        "debug": [False],
    }

    def get_data(self):
        """Load the SVDB dataset."""

        path = fetch_tsb_uad("SVDB")

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        if self.recordings_id in (["all"], "all"):
            self.recordings_id = None
        X, y_true = load_data(path, self.recordings_id, number=self.number)

        X_test = X.copy()
        y_test = y_true.copy()

        X_train = X[:, :int(X.shape[1] * 0.1)]

        if self.debug:
            X_train = X_train[:, :1000]
            X_test = X_test[:, :1000]
            y_test = y_test[:, :1000]

        # Reshaping data to (n_recordings, n_features, n_samples)
        n_recordings = X_train.shape[0]
        X_train = X_train.reshape(n_recordings, 1, -1)
        X_test = X_test.reshape(n_recordings, 1, -1)
        y_test = y_test.reshape(n_recordings, -1)

        plt.figure(figsize=(6, 3))
        plt.plot(X_train[0, 0, :500], linewidth=1.2)
        plt.plot(range(350, 360),
                 X_train[0, 0, 350:360], color="orange", linewidth=3)
        plt.title("SVDB dataset")
        plt.tight_layout()
        plt.savefig("svdb_example.png")
        plt.close()

        print("PLOT SAVED")

        return dict(
            X_train=X_train,
            y_test=y_test,
            X_test=X_test
        )

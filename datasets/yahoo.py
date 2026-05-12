from benchopt import BaseDataset, config

from pathlib import Path
import numpy as np
import pandas as pd

PATH = config.get_data_path("YAHOO")


def load_data(db_path, record_ids=None, verbose=False):
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

    if record_ids is None:
        record_files = list(db_path.glob("*.data.out"))
        record_ids = [f.name for f in record_files]

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Handle case where record_id already includes the pattern
        if record_id.endswith('.data.out'):
            pattern = record_id
        else:
            # Create pattern based on the A{record_id} format
            patterns = [
                f"Yahoo_A{record_id}real_*_data.out",
                f"Yahoo_A{record_id}synthetic_*_data.out",
                f"YahooA{record_id}Benchmark-TS*_data.out"
            ]

        # Find all matching files for this record_id
        matching_files = []
        if record_id.endswith('.data.out'):
            matching_files = list(db_path.glob(pattern))
        else:
            for pattern in patterns:
                matching_files.extend(list(db_path.glob(pattern)))

        if not matching_files:
            if verbose:
                print(f"No files found for record {record_id}")
            continue

        for record_file in matching_files:
            if record_file.exists():
                record_data = pd.read_csv(
                    record_file, header=None).dropna().to_numpy()
                # First column is the data, second column is labels
                if record_data.shape[1] >= 2:
                    data_list.append(record_data[:, 0].astype(float))
                    labels_list.append(record_data[:, 1].astype(int))
                else:
                    if verbose:
                        print(f"Insufficient columns for file {record_file}")
            else:
                if verbose:
                    print(f"Record file not found: {record_file}")

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
    name = "YAHOO"

    parameters = {
        "recordings_id": [["1"]],
        "debug": [False],
    }

    def get_data(self):
        """Load the YAHOO dataset."""

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        X, y_true = load_data(PATH, self.recordings_id)

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

        return dict(
            X_train=X_train,
            y_test=y_test,
            X_test=X_test
        )

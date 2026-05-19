from benchopt import BaseDataset, config

from pathlib import Path
import numpy as np
import pandas as pd

PATH = config.get_data_path("SMD")


def load_data(db_path, record_ids=None):
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
        # Get all available record files matching the pattern
        record_files = list(db_path.glob("machine-*-*.test.csv*"))
        # Extract record IDs from filenames
        record_ids = []
        for f in record_files:
            # Extract from machine-{record_id}-*.test.csv
            parts = f.stem.split('-')
            if len(parts) >= 3:
                record_ids.append(parts[1])
        record_ids = list(set(record_ids))  # Remove duplicates

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Find files matching the pattern
        pattern = f"machine-{record_id}-*.test.csv*"
        record_files = list(db_path.glob(pattern))

        for record_file in record_files:
            if record_file.exists():
                # Load the record data
                record_data = pd.read_csv(
                    record_file, header=None).dropna().to_numpy()
                # Assuming first column is the data, second column is labels
                if record_data.shape[1] >= 2:
                    data_list.append(record_data[:, 0].astype(float))
                    labels_list.append(record_data[:, 1].astype(int))
                else:
                    print(f"Insufficient columns for record {record_id}")
            else:
                print(f"Record file not found: {record_file}")

    if not data_list:
        raise ValueError("No valid data found")

    # Find maximum length for padding
    max_length = max(len(data) for data in data_list)

    # Pad all sequences to the same length
    padded_data = []
    padded_labels = []
    for data, labels in zip(data_list, labels_list):
        if len(data) < max_length:
            # Pad with last value for data and 0 for labels
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
    name = "SMD"

    parameters = {
        "recordings_id": [["1", "2"]],
        "debug": [False],
    }

    def get_data(self):
        """Load the SMD dataset."""

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
        # For SMD, treat as single recording
        n_features = X_train.shape[1]
        X_train = X_train.T.reshape(1, n_features, -1)
        X_test = X_test.T.reshape(1, n_features, -1)
        y_test = y_test.reshape(1, -1)

        return dict(
            X_train=X_train,
            y_test=y_test,
            X_test=X_test
        )

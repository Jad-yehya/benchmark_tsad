from benchopt import BaseDataset, config

from pathlib import Path
import numpy as np
import pandas as pd

PATH = config.get_data_path("IOPS")
PATH = "/data/parietal/store2/data/tsb-uad/TSB-UAD-Public/IOPS/"


def load_data(db_path, verbose=False):
    """
    Load train and test data from the database path.

    Args:
        db_path: Path to the database directory
        verbose: If True, print loading progress information.

    Returns:
        tuple: (X_train, X_test, y_test) where:
            - X_train: nd.array of shape (num_records, num_samples)
            - X_test: nd.array of shape (num_records, num_samples)
            - y_test: nd.array of shape (num_records, num_samples)
    """
    db_path = Path(db_path)

    # Get all train and test files
    train_files = list(db_path.glob("KPI-*.train.out"))
    test_files = list(db_path.glob("KPI-*.test.out"))

    if not train_files or not test_files:
        raise ValueError("No train or test files found")

    # Load train data
    train_data_list = []
    for train_file in train_files:
        record_data = pd.read_csv(train_file, header=None).dropna().to_numpy()
        if record_data.shape[1] >= 1:
            train_data_list.append(record_data[:, 0].astype(float))
        else:
            if verbose:
                print(f"Insufficient columns for train file {train_file}")

    # Load test data and labels
    test_data_list = []
    test_labels_list = []
    for test_file in test_files:
        record_data = pd.read_csv(test_file, header=None).dropna().to_numpy()
        if record_data.shape[1] >= 2:
            test_data_list.append(record_data[:, 0].astype(float))
            test_labels_list.append(record_data[:, 1].astype(int))
        else:
            if verbose:
                print(f"Insufficient columns for test file {test_file}")

    if not train_data_list or not test_data_list:
        raise ValueError("No valid data found")

    # Find maximum length for padding
    max_train_length = max(len(data) for data in train_data_list)
    max_test_length = max(len(data) for data in test_data_list)

    # Pad train sequences
    padded_train_data = []
    for data in train_data_list:
        if len(data) < max_train_length:
            padded_train_data.append(
                np.pad(
                    data,
                    (0, max_train_length - len(data)),
                    mode="constant",
                    constant_values=data[-1],
                )
            )
        else:
            padded_train_data.append(data[:max_train_length])

    # Pad test sequences and labels
    padded_test_data = []
    padded_test_labels = []
    for data, labels in zip(test_data_list, test_labels_list):
        if len(data) < max_test_length:
            padded_test_data.append(
                np.pad(
                    data,
                    (0, max_test_length - len(data)),
                    mode="constant",
                    constant_values=data[-1],
                )
            )
            padded_test_labels.append(
                np.pad(
                    labels,
                    (0, max_test_length - len(labels)),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            padded_test_data.append(data[:max_test_length])
            padded_test_labels.append(labels[:max_test_length])

    return (
        np.array(padded_train_data),
        np.array(padded_test_data),
        np.array(padded_test_labels)
    )


class Dataset(BaseDataset):
    name = "IOPS"

    parameters = {
        "debug": [False],
    }

    def get_data(self):
        """Load the IOPS dataset."""

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        X_train, X_test, y_test = load_data(PATH)

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

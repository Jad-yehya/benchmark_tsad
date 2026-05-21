from benchopt import BaseDataset

from pathlib import Path
import numpy as np
import pandas as pd

from benchmark_utils.download import fetch_tsb_uad


def load_data(db_path, record_ids=None, verbose=False):
    """
    Load data from the database path for specified record IDs.

    Args:
        db_path: Path to the database directory
        record_ids: List of record IDs to load for testing.
        verbose: If True, print loading progress information.

    Returns:
        tuple: (X_train, X_test, y_test) where:
            - X_train: numpy array of shape (num_records, num_samples)
            - X_test: numpy array of shape (num_records, num_samples)
            - y_test: numpy array of shape (num_records, num_samples)
    """
    db_path = Path(db_path)

    # Load training data
    train_files = sorted(list(db_path.glob("room-occupancy.train.csv@*.out")))
    if verbose:
        print(train_files)
    if not train_files:
        raise FileNotFoundError("No training files found.")
    train_data_list = [
        pd.read_csv(f, header=None).dropna().to_numpy()[:, 0].astype(float)
        for f in train_files
    ]
    # Concatenate all training series into a single array
    X_train = np.concatenate(train_data_list)

    # Load testing data
    if record_ids is None:
        record_ids = sorted(
            list(set(
                f.name.split('.')[0].split('-')[-1]
                for f in db_path.glob("room-occupancy-*.test.csv@*.out")
            ))
        )

    test_data_list = []
    labels_list = []
    for record_id in record_ids:
        test_files = sorted(
            list(db_path.glob(f"room-occupancy-{record_id}.test.csv@*.out"))
        )
        if not test_files:
            if verbose:
                print(f"No test files found for record_id {record_id}")
            continue

        for test_file in test_files:
            record_data = pd.read_csv(
                test_file, header=None).dropna().to_numpy()
            if record_data.shape[1] >= 2:
                test_data_list.append(record_data[:, 0].astype(float))
                labels_list.append(record_data[:, 1].astype(int))
            else:
                if verbose:
                    print(
                        f"Insufficient columns "
                        f"for record file {test_file.name}")

    if not test_data_list:
        raise ValueError("No valid test data found")

    # Find maximum length for padding test data
    max_length = max(len(data) for data in test_data_list)

    # Pad all test sequences to the same length
    padded_data = []
    padded_labels = []
    for data, labels in zip(test_data_list, labels_list):
        pad_width = max_length - len(data)
        if pad_width > 0:
            padded_data.append(
                np.pad(
                    data, (
                        0,
                        pad_width),
                    mode="constant",
                    constant_values=data[-1]
                )
            )
            padded_labels.append(
                np.pad(
                    labels, (0, pad_width), mode="constant", constant_values=0
                )
            )
        else:
            padded_data.append(data)
            padded_labels.append(labels)

    X_test = np.array(padded_data)
    y_test = np.array(padded_labels)

    # Reshape X_train to be 2D
    X_train = X_train.reshape(1, -1)

    return X_train, X_test, y_test


class Dataset(BaseDataset):
    name = "OCCUPANCY"

    requirements = ["pip:pooch"]

    parameters = {
        "recordings_id": [None],
        "debug": [False],
    }

    def get_data(self):
        """Load the OCCUPANCY dataset."""

        path = fetch_tsb_uad("OCCUPANCY")

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        X_train, X_test, y_test = load_data(path, self.recordings_id)

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

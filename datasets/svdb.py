from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    from pathlib import Path
    import numpy as np
    import pandas as pd

    PATH = config.get_data_path("SVDB")


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
        record_files = list(db_path.glob("*.test.csv@*.out"))
        record_ids = [f.name for f in record_files]

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Handle case where record_id already includes the pattern
        if record_id.endswith('.test.csv@*.out'):
            pattern = record_id
        else:
            pattern = f"{record_id}.test.csv@*.out"

        # Find all matching files for this record_id
        matching_files = list(db_path.glob(pattern))

        if not matching_files:
            print(f"No files found for record {record_id}")
            continue

        for record_file in matching_files:
            if record_file.exists():
                record_data = pd.read_csv(
                    record_file, header=None).dropna().to_numpy()
                # Assuming first column is the data, second column is labels
                if record_data.shape[1] >= 2:
                    data_list.append(record_data[:, 0].astype(float))
                    labels_list.append(record_data[:, 1].astype(int))
                else:
                    print(f"Insufficient columns for file {record_file}")
            else:
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
    name = "SVDB"

    parameters = {
        "recordings_id": [["801"]],
        "debug": [False],
    }

    def get_data(self):
        """Load the SVDB dataset."""

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

        # Reshaping data to (n_samples, n_features)
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return dict(
            X_train=X_train,
            y_test=y_test,
            X_test=X_test
        )

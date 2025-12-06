from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    from pathlib import Path
    import numpy as np
    import pandas as pd

    PATH = config.get_data_path("MITDB")


def load_mitdb_data(db_path, record_ids=None, verbose=False):
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
        # Get all available record files with format like 100.test.csv@1.out
        record_files = list(db_path.glob("*.out"))
        record_ids = [str(f.name).split(".")[0] for f in record_files]

    if verbose:
        print(f"Loading records: {record_ids}")

    data_list = []
    labels_list = []
    for record_id in record_ids:
        # Find file starting with record_id and ending with .out
        record_files = list(db_path.glob(f"{record_id}*.out"))
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
    name = "MITDB"

    parameters = {
        "recordings_id": [["100", "201", "109", "105", "111", "221"]],
        "debug": [False],
    }

    def get_data(self):
        """Load the MITDB dataset."""

        # X shape (n_recordings, n_samples)
        # y shape (n_recordings, n_samples)
        if self.recordings_id in (["all"], "all"):
            self.recordings_id = None
        X, y_true = load_mitdb_data(PATH, self.recordings_id)

        X_test = X.copy()
        y_test = y_true.copy()

        X_train = X[:, : int(X.shape[1] * 0.1)]

        if self.debug:
            X_train = X_train[:, -2000:]
            X_test = X_test[:, -2000:]
            y_test = y_test[:, -2000:]

        # Reshaping data to (n_recordings, n_features, n_samples)
        n_recordings = X.shape[0]
        X_train = X_train.reshape(n_recordings, 1, -1)
        X_test = X_test.reshape(n_recordings, 1, -1)
        y_test = y_test.reshape(n_recordings, -1)

        return dict(X_train=X_train, y_test=y_test, X_test=X_test)

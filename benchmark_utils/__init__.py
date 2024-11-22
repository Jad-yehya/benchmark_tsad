# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from benchopt import safe_import_context
import os

with safe_import_context() as import_ctx:
    import numpy as np


def mean_overlaping_pred(predictions, stride):
    """
    Averages overlapping predictions for multivariate time series.

    Args:
    predictions: np.ndarray, shape (n_windows, H, n_features)
                The predicted values for each window for each feature.
    stride: int
            The stride size.

    Returns:
    np.ndarray: Averaged predictions for each feature.
    """
    n_windows, H, n_features = predictions.shape
    total_length = (n_windows-1) * stride + H - 1

    # Array to store accumulated predictions for each feature
    accumulated = np.zeros((total_length, n_features))
    # store the count of predictions at each point for each feature
    counts = np.zeros((total_length, n_features))

    # Accumulate predictions and counts based on stride
    for i in range(n_windows):
        start = i * stride
        accumulated[start:start+H] += predictions[i]
        counts[start:start+H] += 1

    # Avoid division by zero
    counts[counts == 0] = 1

    # Average the accumulated predictions
    averaged_predictions = accumulated / counts

    return averaged_predictions


def check_data(data_path, dataset, data_type):
    """
    Checks if the data is present in the specified path.

    Args:
    data_path: str
               The path to the data directory.
    dataset: str
             The name of the dataset, either 'WADI' or 'SWaT'.
    data_type: str
               The type of data, either 'train' or 'test'.

    Raises:
    ImportError: If the required data files are not found.
    """
    if dataset == "WADI":
        if data_type == "train":
            required_files = ["WADI_14days_new.csv"]
        elif data_type == "test":
            required_files = ["WADI_attackdataLABLE.csv"]
        else:
            raise ValueError("data_type must be either 'train' or 'test'")
    elif dataset == "SWaT":
        if data_type == "train":
            required_files = ["swat_train2.csv"]
        elif data_type == "test":
            required_files = ["swat2.csv"]
        else:
            raise ValueError("data_type must be either 'train' or 'test'")
    else:
        raise ValueError("dataset must be either 'WADI' or 'SWaT'")

    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            official_repo = {
                "WADI": "https://itrust.sutd.edu.sg/itrust-labs_datasets/\
                    dataset_info/",
                "SWaT": "https://drive.google.com/drive/folders/\
                    1xhcYqh6okRs98QJomFWBKNLw4d1T4Q0w"
            }
            raise ImportError(
                f"{data_type.capitalize()} data not found for {dataset}. "
                "Please download the data "
                "from the official repository "
                f"{official_repo[dataset]}"
                f"and place it in {data_path}"
            )

from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path
from benchmark_utils import check_data

with safe_import_context() as import_ctx:
    import pandas as pd

    # Checking if the data is available
    PATH = get_data_path(key="SWaT")
    TRAIN_PATH = check_data(PATH, "SWaT", "train")
    TEST_PATH = check_data(PATH, "SWaT", "test")


class Dataset(BaseDataset):
    name = "SWaT"

    parameters = {
        "debug": [False],
    }

    test_parameters = {
        "debug": [True],
    }

    def get_data(self):
        # To get the data, you need to ask for access to the dataset
        # at the following link:
        # https://drive.google.com/drive/folders/1xhcYqh6okRs98QJomFWBKNLw4d1T4Q0w

        # Load the data
        X_train = pd.read_csv(PATH / TRAIN_PATH)
        X_test = pd.read_csv(PATH / TEST_PATH)

        # Extract the target
        y_test = X_test["Normal/Attack"].values
        X_test = X_test.drop(columns=["Normal/Attack"])
        X_test = X_test.to_numpy()

        X_train = X_train.drop(columns=["Normal/Attack"])
        X_train = X_train.to_numpy()

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )

from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import pandas as pd


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

        path = get_data_path(key="SWaT")

        if not (path / "swat_train2.csv").exists():
            raise FileNotFoundError(
                "Train data not found. Please download the data "
                "from the Google Drive "
                "https://drive.google.com/drive/folders/"
                "1xhcYqh6okRs98QJomFWBKNLw4d1T4Q0w"
                f" and place it in {path}"
            )

        if not (path / "swat2.csv").exists():
            raise FileNotFoundError(
                "Test data not found. Please download the data "
                "from the Google Drive "
                "https://drive.google.com/drive/folders/"
                "1xhcYqh6okRs98QJomFWBKNLw4d1T4Q0w"
                f" and place it in {path}"
            )

        # Load the data
        X_train = pd.read_csv(path / "swat_train2.csv")
        X_test = pd.read_csv(path / "swat2.csv")

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

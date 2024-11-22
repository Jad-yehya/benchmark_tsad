from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import numpy as np
    import requests

# Create global variables to store the urls
URL_XTRAIN = (
    "https://drive.google.com/uc?&id="
    "1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa&export=download"
)

URL_XTEST = (
    "https://drive.google.com/uc?&id="
    "1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM&export=download"
)

URL_YTEST = (
    "https://drive.google.com/uc?&id="
    "19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd&export=download"
)


class Dataset(BaseDataset):
    name = "MSL"

    parameters = {
        "debug": [False],
    }

    test_parameters = {
        "debug": [True],
    }

    def get_data(self):
        path = config.get_data_path(key="MSL")
        # Check if the data is already here
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

            response = requests.get(URL_XTRAIN)
            with open(path / "MSL_train.npy", "wb") as f:
                f.write(response.content)
            response = requests.get(URL_XTEST)
            with open(path / "MSL_test.npy", "wb") as f:
                f.write(response.content)
            response = requests.get(URL_YTEST)
            with open(path / "MSL_test_label.npy", "wb") as f:
                f.write(response.content)

        X_train = np.load(path / "MSL_train.npy", allow_pickle=True)
        X_test = np.load(path / "MSL_test.npy", allow_pickle=True)
        y_test = np.load(path / "MSL_test_label.npy", allow_pickle=True)

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        print(X_train.shape, X_test.shape, y_test.shape)

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )

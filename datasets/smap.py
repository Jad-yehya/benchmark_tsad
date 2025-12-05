from benchopt import BaseDataset, safe_import_context, config

with safe_import_context() as import_ctx:
    import numpy as np
    import requests
    # from sklearn.model_selection import TimeSeriesSplit

URL_XTRAIN = (
    "https://drive.google.com/uc?&id=1e_JhpIURD"
    "Lluw4IcHJF-dgtjjJXsPEKE&export=download"
)
URL_XTEST = (
    "https://drive.google.com/uc?&id=10"
    "-r-Zm0nfQJp0i-mVg3iXs6x0u9Ua25a&export=download"
)
URL_YTEST = (
    "https://drive.google.com/uc?&id=1uYiXqmK3C"
    "gyxk4U6-LgUni7JddQnlggs&export=download"
)


class Dataset(BaseDataset):
    name = "SMAP"

    parameters = {
        "debug": [False],
        "n_splits": [5],
        "validation_size": [0.2],
    }

    test_parameters = {
        "debug": [True],
        "n_splits": [2],
        "validation_size": [0.2],
    }

    def get_data(self):
        path = config.get_data_path(key="SMAP")

        # Check if the data is already here
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

            response = requests.get(URL_XTRAIN)
            with open(path / "SMAP_train.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(URL_XTEST)
            with open(path / "SMAP_test.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(URL_YTEST)
            with open(path / "SMAP_test_label.npy", "wb") as f:
                f.write(response.content)

        X_train = np.load(path / "SMAP_train.npy")
        X_test = np.load(path / "SMAP_test.npy")
        y_test = np.load(path / "SMAP_test_label.npy")

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        # Reshaping data to (n_recordings, n_features, n_samples)
        # For SMAP, treat as single recording
        n_features = X_train.shape[1]
        X_train = X_train.T.reshape(1, n_features, -1)
        X_test = X_test.T.reshape(1, n_features, -1)
        y_test = y_test.reshape(1, -1)

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )

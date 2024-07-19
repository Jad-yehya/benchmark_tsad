from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np
    import requests

url_xtrain = (
    "https://drive.google.com/uc?&id=1qCi9dXN"
    "idRvQ6haGbmjGxwSDn17KWSha&export=download"
)
url_xtest = (
    "https://drive.google.com/uc?&id=16oiQRvH"
    "-0qYDBKj2FRIRnWbWGdfwXyOZ&export=download"
)
url_ytest = (
    "https://drive.google.com/uc?&id=1ezC01a0"
    "74tlYa-3nnwu49ywmFuibfaQh&export=download"
)


class Dataset(BaseDataset):
    name = "GECCO"

    install_cmd = "conda"
    requirements = ["pandas"]

    parameters = {
        "debug": [False],
    }

    def get_data(self):

        # Check if the data is already here
        if not os.path.exists("data/GECCO/GECCO_train.npy"):
            os.makedirs("data/GECCO", exist_ok=True)

            response = requests.get(url_xtrain)
            with open("data/GECCO/GECCO_train.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_xtest)
            with open("data/GECCO/GECCO_test.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_ytest)
            with open("data/GECCO/GECCO_test_label.npy", "wb") as f:
                f.write(response.content)

        X_train = np.load("data/GECCO/GECCO_train.npy")
        X_test = np.load("data/GECCO/GECCO_test.npy")
        y_test = np.load("data/GECCO/GECCO_test_label.npy")

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )

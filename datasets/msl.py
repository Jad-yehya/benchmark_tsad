from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import os
    import numpy as np
    import requests

# Create global variables to store the urls
url_xtrain = (
    "https://drive.google.com/uc?&id="
    "1PMzjODVFblVnwq8xo7pKHrdbczPxdqTa&export=download"
)

url_xtest = (
    "https://drive.google.com/uc?&id="
    "1OcNc0YQsOMw9jQIIHgiOXVG03wjXbEiM&export=download"
)

url_ytest = (
    "https://drive.google.com/uc?&id="
    "19vR0QvKluuiIT2H5mCFNIJh6xGVwshDd&export=download"
)


class Dataset(BaseDataset):
    name = "MSL"

    install_cmd = "conda"
    requirements = ["pandas", "requests"]

    parameters = {
        "debug": [False],
    }

    def get_data(self):
        # Adding get_data_path method soon

        # Check if the data is already here
        if not os.path.exists("data/MSL/MSL_train.npy"):
            os.makedirs("data/MSL", exist_ok=True)

            response = requests.get(url_xtrain)
            with open("data/MSL/MSL_train.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_xtest)
            with open("data/MSL/MSL_test.npy", "wb") as f:
                f.write(response.content)

            response = requests.get(url_ytest)
            with open("data/MSL/MSL_test_label.npy", "wb") as f:
                f.write(response.content)

        X_train = np.load("data/MSL/MSL_train.npy")
        X_test = np.load("data/MSL/MSL_test.npy")
        y_test = np.load("data/MSL/MSL_test_label.npy")

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        print(X_train.shape, X_test.shape, y_test.shape)

        return dict(
            X_train=X_train, y_test=y_test, X_test=X_test
        )

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from rosecdl.utils.utils_signal import generate_experiment


class Dataset(BaseDataset):
    name = "Pattern"

    parameters = {
        "n_samples": [10],
        "n_times": [5000],
        "debug": [False],
        "random_state": [42],
        "n_times_atom": [250],
    }


    def get_data(self):
        if self.debug:
            self.n_samples = 2
            self.n_times = 1000

        size = self.n_times // 5000
        contamination_params = {
                "n_atoms": 2,
                "sparsity": 3,
                "init_z": "constant",
                "init_z_kwargs": {"value": 50},
            }

        simulation_params = {
            "n_trials": self.n_samples * 2,
            "n_channels": 2,
            "n_times": self.n_times,
            "n_atoms": 2,
            "n_times_atom": self.n_times_atom,
            "n_atoms_extra": 2,  # extra atoms in the learned dictionary
            "D_init": "random",
            "window": True,
            "contamination_params": contamination_params,
            "init_d": "shapes",
            "init_d_kwargs": {"shapes": ["sin", "gaussian"]},
            "init_z": "constant",
            "init_z_kwargs": {"value": 1},
            "noise_std": 0.01,
            "rng": self.random_state,
            "sparsity": 20,
        }

        X, _, _, _, info_contam = generate_experiment(
            simulation_params=simulation_params,
            return_info_contam=True,
        )

        X_train, X_test = X[: self.n_samples], X[self.n_samples :]
        y_test = info_contam["outliers_mask"][self.n_samples :]
        y_test = np.any(y_test, axis=1)


        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        return dict(X_train=X_train, y_test=y_test, X_test=X_test)

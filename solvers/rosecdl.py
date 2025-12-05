from benchopt import safe_import_context, BaseSolver

with safe_import_context() as import_ctx:
    from rosecdl.rosecdl import RoseCDL
    from TSB_AD.utils.slidingWindows import find_length
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime


class Solver(BaseSolver):
    name = "RoseCDL"

    install_cmd = "conda"
    requirements = ["pip:rosecdl", "pip:torch"]

    parameters = {
        "n_components": [1],
        "kernel_size": ["auto"],
        "lmbd": [0.8],
        "scale_lmbd": [False],
        "epochs": [70],
        "max_batch": [None],
        "mini_batch_size": [600],
        "sample_window": [1_000],
        "optimizer": ["linesearch"],
        "n_iterations": [90],
        "window": [False],
        "outliers_kwargs": [
            {
                "method": "mad",
                "alpha": 3.5,
                "moving_average": None,
                "union_channels": True,
                "opening_window": True,
            },
        ],
        "plot": [False],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X_train, y_test, X_test):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # We receive data in shape (n_recordings, n_features, n_samples)
        self.y_test = y_test
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.X_test = X_test

        if self.kernel_size == "auto":
            self.kernel_size = int(find_length(X_train.reshape(-1)))

        print("=====================")
        print(f"kernel_size: {self.kernel_size}")
        print("=====================")

        self.clf = RoseCDL(
            n_components=self.n_components,
            n_channels=X_train.shape[1],
            kernel_size=self.kernel_size,
            lmbd=self.lmbd,
            scale_lmbd=self.scale_lmbd,
            epochs=self.epochs,
            max_batch=self.max_batch,
            mini_batch_size=self.mini_batch_size,
            sample_window=self.sample_window,
            optimizer=self.optimizer,
            n_iterations=self.n_iterations,
            window=self.window,
            device=self.device,
            outliers_kwargs=self.outliers_kwargs,
        )

    def run(self, _):
        self.clf.fit(self.X_train)
        del self.X_train  # Free GPU memory for X_train after fitting
        self.y_pred = self.clf.get_outlier_mask(self.X_test)

        xh, zh = self.clf.csc(
            torch.tensor(self.X_test, dtype=torch.float32, device=self.device)
        )
        err = self.clf.loss_fn.compute_patch_error(
            X_hat=xh,
            z_hat=zh,
            X=torch.tensor(self.X_test, dtype=torch.float32, device=self.device),
        )
        err = err.cpu().detach().numpy()
        # Aggregate errors over channels
        self.err = err.sum(axis=1).reshape(-1)
        del self.clf  # Free GPU memory for the model
        torch.cuda.empty_cache()  # Release cached GPU memory

    def _plot_anomalies(self):
        y_test_flat = self.y_test.flatten()
        y_pred_flat = self.y_pred.flatten()

        dataset_name = str(self._objective._dataset).split("[")[0]

        true_positives_indices = np.where((y_test_flat == 1) & (y_pred_flat == 1))[0]
        false_negatives_indices = np.where((y_test_flat == 1) & (y_pred_flat == 0))[0]

        if isinstance(self.X_test, torch.Tensor):
            X_test_numpy = self.X_test.cpu().numpy()
        else:
            X_test_numpy = self.X_test

        # Select the first recording and squeeze to get (n_samples,)
        X_test_squeezed = X_test_numpy[0].squeeze()

        from tueplots import bundles

        plt.rcParams.update(bundles.aistats2025())

        # use no tex
        plt.rcParams.update({"text.usetex": False})

        plot_window_size = 1000
        min_overlap_ratio = 0.1
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        max_plots = 5

        # Plot up to 5 true positive examples with minimum overlap
        if true_positives_indices.size > 0:
            tp_plots_count = 0
            plotted_tp_anomalies = set()
            for tp_idx in true_positives_indices:
                if tp_plots_count >= max_plots:
                    break

                # Find the bounds of the true anomaly containing this tp_idx
                true_start = tp_idx
                while true_start > 0 and y_test_flat[true_start - 1] == 1:
                    true_start -= 1
                true_end = tp_idx
                while (
                    true_end < len(y_test_flat) - 1 and y_test_flat[true_end + 1] == 1
                ):
                    true_end += 1

                if (true_start, true_end) in plotted_tp_anomalies:
                    continue

                true_len = true_end - true_start + 1
                # Calculate overlap
                overlap_indices = np.where(
                    (y_test_flat[true_start : true_end + 1] == 1)
                    & (y_pred_flat[true_start : true_end + 1] == 1)
                )[0]
                overlap_len = len(overlap_indices)

                if true_len > 0 and (overlap_len / true_len) >= min_overlap_ratio:
                    plotted_tp_anomalies.add((true_start, true_end))
                    start = max(0, tp_idx - plot_window_size // 2)
                    end = min(X_test_squeezed.shape[0], tp_idx + plot_window_size // 2)

                    plt.figure(figsize=(3.25, 2))
                    plt.plot(np.arange(start, end), X_test_squeezed[start:end])
                    plt.title(
                        f"RoseCDL Successful Detection\n{dataset_name} dataset"
                    )
                    plt.xlabel("Time")
                    plt.ylabel("Value")

                    true_anomaly_indices = np.where(y_test_flat[start:end] == 1)[0]
                    if true_anomaly_indices.size > 0:
                        plt.axvspan(
                            start + true_anomaly_indices[0],
                            start + true_anomaly_indices[-1],
                            color="yellow",
                            alpha=0.5,
                            label="True Anomaly",
                        )

                    pred_anomaly_indices = np.where(y_pred_flat[start:end] == 1)[0]
                    if pred_anomaly_indices.size > 0:
                        plt.axvspan(
                            start + pred_anomaly_indices[0],
                            start + pred_anomaly_indices[-1],
                            color="red",
                            alpha=0.3,
                            label="Predicted Anomaly",
                        )

                    plt.legend()
                    plt.savefig(
                        f"anomaly_examples/{dataset_name.lower()}/rosecdl_successful_detection_{dataset_name.lower()}_{timestamp}_{tp_plots_count}.pdf",
                        format="pdf",
                    )
                    plt.close()
                    tp_plots_count += 1

            if tp_plots_count == 0:
                print("Could not find a true positive with sufficient overlap to plot.")

        # Plot up to 5 false negative examples
        if false_negatives_indices.size > 0:
            fn_plots_count = 0
            plotted_fn_anomalies = set()
            for fn_idx in false_negatives_indices:
                if fn_plots_count >= max_plots:
                    break

                # Find the bounds of the true anomaly containing this fn_idx
                true_start = fn_idx
                while true_start > 0 and y_test_flat[true_start - 1] == 1:
                    true_start -= 1
                true_end = fn_idx
                while (
                    true_end < len(y_test_flat) - 1 and y_test_flat[true_end + 1] == 1
                ):
                    true_end += 1

                if (true_start, true_end) in plotted_fn_anomalies:
                    continue

                plotted_fn_anomalies.add((true_start, true_end))
                start = max(0, fn_idx - plot_window_size // 2)
                end = min(X_test_squeezed.shape[0], fn_idx + plot_window_size // 2)

                plt.figure(figsize=(3.25, 2))
                plt.plot(np.arange(start, end), X_test_squeezed[start:end])
                plt.title(
                    f"RoseCDL Failed Detection\n{dataset_name} dataset"
                )
                plt.xlabel("Time")
                plt.ylabel("Value")

                true_anomaly_indices = np.where(y_test_flat[start:end] == 1)[0]
                if true_anomaly_indices.size > 0:
                    plt.axvspan(
                        start + true_anomaly_indices[0],
                        start + true_anomaly_indices[-1],
                        color="yellow",
                        alpha=0.5,
                        label="True Anomaly (missed)",
                    )

                plt.legend()
                plt.savefig(
                    f"anomaly_examples/{dataset_name.lower()}/rosecdl_failed_detection_{dataset_name.lower()}_{timestamp}_{fn_plots_count}.pdf",
                    format="pdf",
                )
                plt.close()
                fn_plots_count += 1

    def get_result(self):
        if self.plot:
            self._plot_anomalies()

        return dict(y_hat=self.y_pred, raw_anomaly_score=self.err)

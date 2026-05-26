# Benchmark repository for Time Series Anomaly Detection (TSAD) algorithms

![Build Status](https://github.com/benchopt/benchmark_tsad/workflows/Tests/badge.svg)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

This repository is a Benchopt benchmark for unsupervised time series anomaly
detection. It compares classical detectors, reconstruction models, forecasting
models, matrix-profile methods, dictionary-learning methods, and foundation
model baselines.

## Quick Start

Install Benchopt and clone the benchmark:

```bash
pip install -U benchopt
git clone https://github.com/Jad-yehya/benchmark_tsad
cd benchmark_tsad
```

Run a small local experiment:

```bash
benchopt run . -s RoseCDL -d Simulated
```

Run a specific method on a benchmark dataset:

```bash
benchopt run . -s MP -d SMAP
```

Show all Benchopt options:

```bash
benchopt run . --help
```

## Benchmark Interface

Datasets return:

```python
dict(
    X_train=X_train,  # shape: (n_recordings, n_features, n_samples)
    X_test=X_test,    # shape: (n_recordings, n_features, n_samples)
    y_test=y_test,    # shape: (n_recordings, n_samples)
)
```

Solvers receive only `X_train` and `X_test`. They return continuous anomaly
scores by default:

```python
return dict(anomaly_scores=anomaly_scores)
```

Larger scores must mean "more anomalous".

Solvers may optionally return binary predictions:

```python
return dict(
    anomaly_scores=anomaly_scores,
    anomaly_predictions=anomaly_predictions,
)
```

Prediction generation is solver-owned. Solvers that support prediction metrics
expose:

- `cutoff`: fraction of the highest test scores to label as anomalous.

For example, `cutoff=0.05` means the top 5% highest anomaly scores are returned
as `anomaly_predictions = 1`. If `cutoff` is not set, the solver returns only
scores.

## Models

| Solver name | Method family | Main idea | Notes |
| --- | --- | --- | --- |
| `ABOD` | Classical outlier detection | Angle-Based Outlier Detection on windowed features | PyOD; legacy solver |
| `AE` | Neural reconstruction | Feed-forward autoencoder on sliding windows | PyTorch |
| `AnomalyBERT` | Transformer | Anomaly Transformer-style masked replacement objective | Requires local AnomalyBERT checkout |
| `AR` | Forecasting | Autoregressive linear model with reconstruction error | PyTorch |
| `CBLOF` | Classical outlier detection | Cluster-Based Local Outlier Factor on windowed features | PyOD; legacy solver |
| `DAGMM` | Deep density model | Deep Autoencoding Gaussian Mixture Model | Merlion |
| `DIF` | Deep ensemble outlier detection | Deep Isolation Forest | PyOD; GPU recommended |
| `IsolationForest` | Classical outlier detection | Isolation Forest on raw/windowed features | scikit-learn; legacy solver |
| `LocalOutlierFactor` | Classical outlier detection | Local density deviation with novelty mode | scikit-learn; legacy solver |
| `LSTM` | Sequence reconstruction | LSTM encoder-decoder reconstruction error | PyTorch |
| `MP` | Matrix profile | Distance profile over subsequences | TSB-AD; univariate only |
| `OCSVM` | Classical outlier detection | One-Class SVM on windowed features | scikit-learn; legacy solver |
| `RoseCDL` | Dictionary learning | Convolutional dictionary learning reconstruction error | RoseCDL |
| `Transformer` | Sequence reconstruction | Vanilla transformer reconstruction error | PyTorch |
| `TSB-Chronos` | Foundation model / forecasting | Chronos wrapper from TSB-AD | optional backend |
| `TSB-TimesFM` | Foundation model / forecasting | TimesFM wrapper from TSB-AD | optional backend |
| `TSB-TimesNet` | Deep forecasting | TimesNet wrapper from TSB-AD | optional backend |
| `VAE` | Neural reconstruction | Variational Autoencoder on windows | PyOD |

Some solvers depend on optional or heavy packages. Benchopt installs solver
dependencies in isolated environments when possible. The validation suite marks
optional backends and unavailable local checkouts as expected skips or xfails.

## Datasets

| Dataset name | Type |
| --- | --- |
| `DAPHNET` | Real sensor/health time series |
| `DODGERS` | Real event-count time series |
| `ECG` | Real ECG time series |
| `GENESIS` | Real anomaly-detection benchmark data |
| `GHL` | Real sensor benchmark data |
| `IOPS` | Real server metric time series |
| `KDD21` | Real anomaly-detection benchmark data |
| `MGAB` | Real anomaly-detection benchmark data |
| `MITDB` | Real ECG time series |
| `MSL` | NASA Mars Science Laboratory telemetry |
| `NAB` | Numenta Anomaly Benchmark data |
| `OCCUPANCY` | Real occupancy/environmental time series |
| `OPPORTUNITY` | Real sensor/activity time series |
| `Pattern` | Synthetic convolutional-pattern anomalies |
| `PSM` | Pooled Server Metrics |
| `SENSORSCOPE` | Real environmental sensor time series |
| `Simulated` | Synthetic regression time series with injected anomalies |
| `SMAP` | NASA Soil Moisture Active Passive telemetry |
| `SMD` | Server Machine Dataset |
| `SVDB` | Real ECG time series |
| `SWaT` | Secure Water Treatment industrial process |
| `Trend` | Synthetic pattern anomalies with low-frequency trend |
| `WADI` | Water Distribution industrial process |
| `YAHOO` | Yahoo anomaly benchmark data |

`SMAP`, `MSL`, and `PSM` are fetched automatically when missing. `Simulated`,
`Pattern`, and `Trend` are generated at runtime.

Most remaining loaders expect files under Benchopt's data directory, available
through `benchopt.config.get_data_path(<DATASET_NAME>)`.

`SWaT` and `WADI`
cannot be redistributed; request access from the dataset owners:
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/.

## Metrics

The objective separates score-based metrics from prediction-based metrics.

### Score Metrics

These are computed from `anomaly_scores` and are enabled by default.

| Metric | Key | Direction |
| --- | --- | --- |
| Area under the precision-recall curve | `auc_pr` | higher is better |
| Area under the ROC curve | `auc_roc` | higher is better |

Default objective configuration:

```python
score_metrics = ("auc_pr", "auc_roc")
prediction_metrics = None
```

### Prediction Metrics

These are computed from `anomaly_predictions`. They are opt-in and require a
solver `cutoff` or a solver that otherwise returns predictions.

| Metric | Key | Direction |
| --- | --- | --- |
| Point-wise precision | `precision` | higher is better |
| Point-wise recall | `recall` | higher is better |
| Point-wise F1 | `f1` | higher is better |
| Zero-one loss | `zoloss` | lower is better |
| Range precision | `precision_t` | higher is better |
| Range recall | `recall_t` | higher is better |
| Range F1 | `f1_t` | higher is better |
| Candidate-to-target temporal distance | `ctt` | lower is better |
| Target-to-candidate temporal distance | `ttc` | lower is better |
| Time-forgiving precision | `soft_precision_{1,3,5,10,20}` | higher is better |
| Time-forgiving recall | `soft_recall_{1,3,5,10,20}` | higher is better |
| Time-forgiving F1 | `soft_f1_{1,3,5,10,20}` | higher is better |

Prediction metric aliases:

- `prediction_metrics="all"` expands to all prediction metrics.
- `soft_precision`, `soft_recall`, and `soft_f1` expand over detection ranges
  `1, 3, 5, 10, 20`.

Benchopt expects a scalar `value` in every result row. The benchmark reports all
requested metrics and uses a finite fallback value when no primary optimization
metric is selected.

## Development

Run the repository tests:

```bash
python -m pytest
```

Run Benchopt's benchmark validation suite:

```bash
benchopt test . -- -q
```

Run a quick syntax check:

```bash
python -m py_compile objective.py benchmark_utils/*.py solvers/*.py solvers/legacy/*.py
```

Useful repository checks before opening a pull request:

```bash
git diff --check
python -m pytest
benchopt test . -- -q -x
```

## Adding a Solver

New solvers should follow the score-first convention:

```python
class Solver(BaseSolver):
    name = "MySolver"

    parameters = {
        "cutoff": [None],
    }

    def set_objective(self, X_train, X_test):
        ...

    def run(self, _):
        self.anomaly_scores = ...
        self.anomaly_predictions = cutoff_scores(
            self.anomaly_scores,
            cutoff=self.cutoff,
        )

    def get_result(self):
        result = dict(anomaly_scores=self.anomaly_scores)
        if self.anomaly_predictions is not None:
            result["anomaly_predictions"] = self.anomaly_predictions
        return result
```

Keep scores oriented so larger means more anomalous. Use `np.nan` for unscored
score positions and `-1` for ignored prediction positions.

## Contributing

Contributions are welcome. Keep changes reproducible and scoped:

- preserve the canonical `anomaly_scores` and optional `anomaly_predictions`
  output format;
- add focused tests for objective behavior, solver output shapes, and metric
  regressions;
- avoid committing downloaded datasets, model checkpoints, generated outputs,
  caches, or local environment files;
- document optional dependencies and expected skips clearly.

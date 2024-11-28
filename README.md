# Benchmark repository for Time Series Anomaly Detection (TSAD) algorithms

![Build Status](https://github.com/Jad-yehya/benchmark_tsad/workflows/Tests/badge.svg)

## About Benchopt

Benchopt is a package to simplify and make more transparent and reproducible the comparisons of optimization algorithms.


## Objective

This benchmark evaluates and compares Time Series Anomaly Detection (TSAD) algorithms. The goal is to:

- Provide a standardized framework for comparing different TSAD approaches
- Enable reproducible evaluation of anomaly detection performance

## Install

This benchmark can be run using the following command:

```bash
$ pip install -U benchopt
$ git clone https://github.com/Jad-yehya/benchmark_tsad
$ benchopt run benchmark_tsad
```
Options for running the benchmark can be passed as command line arguments.
For example, to run a specific solver and a specific dataset, use the following command:

```bash
$ benchopt run benchmark_tsad -s IsolationForest -d SMAP
```

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

## Available Solvers

- AR (Autoregressive Linear Model)
- ABOD (Angle-Based Outlier Detection)
- CBLOF (Cluster-Based Local Outlier Factor)
- DIF (Deep Isolation Forest)
- Isolation Forest
- LOF (Local Outlier Factor)
- LSTM (Long Short-Term Memory)
- OCSVM (One-Class SVM)
- VAE (Variational Autoencoder)
- Transformer

## Datasets

- Soil Moisture Active Passive (SMAP)
- Mars Science Laboratory (MSL)
- Pooled Server Metric (PSM)
- Secure Water Treatment (SWaT)
- Water Distribution (WADI.A2_19)
- Simulated dataset

The SMAP, MSL and PSM datasets are automatically fetched when running the benchmark. 
The simulated dataset is generated at running time. 
However, the automatic use of the SWaT and WADI datasets is not possible. In order to use them, you must request access to the owners at the following link :
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/. 

## Metrics

The benchmark uses three categories of metrics to evaluate the performance of unsupervised time series anomaly detection methods:

1. **Classical Binary Classification Metrics.**  These traditional metrics treat anomaly detection as a point-wise binary classification task.
    - Precision: The ratio of correctly identified anomalies to all predicted anomalies
    - Recall: The ratio of correctly identified anomalies to all actual anomalies
    - F1-Score: The harmonic mean of precision and recall

2. **Time-Forgiving Metrics.** These metrics introduce temporal flexibility by allowing predictions within a certain time window of the actual anomaly to be considered correct.
    - Time-Forgiving Precision (soft_precision_s): The ratio of predictions that fall within the acceptable time window of true anomalies of size *s*.
    - Time-Forgiving Recall (soft_recall_s): The ratio of true anomalies that have at least one prediction within a certain time window of size *s*.
    - Time-Forgiving F1-Score (soft_f1_s): The harmonic mean of the time-forgiving precision and recall.

    - Range_metrics : Another family of precision, recall and f1-score tailored for time series anomaly detection. (Noted precision_t, recall_t, f1_t).

3. **Temporal Distance Metrics.**
These metrics quantify the temporal offset between predicted and actual anomalies, providing insights into whether the detection method tends to identify anomalies early or late. (CTT and TTC).



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

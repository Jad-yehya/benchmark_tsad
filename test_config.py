import sys  # noqa: F401

import pytest  # noqa: F401

from benchopt.utils.sys_info import get_cuda_version


def check_test_solver_install(benchmark, solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name.lower() == "dif":
        if get_cuda_version() is None:
            pytest.xfail("Deep IsolationForest needs a working GPU hardware.")

    if solver_class.name.lower() == "anomalybert":
        pytest.xfail("AnomalyBERT needs to be installed locally from repo"
                     " at https://github.com/Jhryu30/AnomalyBERT.git")

    # if solver_class.name.lower() == "lstm":
    #     if get_cuda_version() is None:
    #         pytest.xfail("LSTM needs a working GPU hardware.")

    # if solver_class.name.lower() == "transformer":
    #     if get_cuda_version() is None:
    #         pytest.xfail("Transformer needs a working GPU hardware.")


def check_test_dataset_get_data(benchmark, data_class):
    if data_class.name.lower() in [
        "daphnet", "dodgers", "ecg", "genesis", "ghl",
        "iops", "kdd21", "mgab", "mitdb", "nab",
        "occupancy", "opportunity", "sensorscope", "smd",
        "svdb", "yahoo"
    ]:
        pytest.xfail(f"{data_class.name} dataset is not downloaded.")

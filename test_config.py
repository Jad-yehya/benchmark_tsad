import sys  # noqa: F401
from importlib.util import find_spec

import pytest  # noqa: F401

from benchopt.utils.sys_info import get_cuda_version


OPTIONAL_BACKEND_INSTALL_XFAILS = {
    "dagmm": "DAGMM depends on the optional salesforce-merlion package.",
    "mp": "MP depends on the optional TSB-AD package.",
    "rosecdl": "RoseCDL depends on an optional GitHub package.",
    "tsb-chronos": "TSB-Chronos depends on the optional TSB-AD backend.",
    "tsb-timesfm": "TSB-TimesFM depends on TSB-AD and timesfm.",
    "tsb-timesnet": "TSB-TimesNet depends on the optional TSB-AD backend.",
}


def check_test_solver_install(benchmark, solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    solver_name = solver_class.name.lower()

    if solver_name in OPTIONAL_BACKEND_INSTALL_XFAILS:
        pytest.xfail(OPTIONAL_BACKEND_INSTALL_XFAILS[solver_name])

    if solver_name == "dif":
        if get_cuda_version() is None:
            pytest.xfail("Deep IsolationForest needs a working GPU hardware.")

    if solver_name == "anomalybert":
        pytest.xfail("AnomalyBERT needs to be installed locally from repo"
                     " at https://github.com/Jhryu30/AnomalyBERT.git")

    # if solver_class.name.lower() == "lstm":
    #     if get_cuda_version() is None:
    #         pytest.xfail("LSTM needs a working GPU hardware.")

    # if solver_class.name.lower() == "transformer":
    #     if get_cuda_version() is None:
    #         pytest.xfail("Transformer needs a working GPU hardware.")


def check_test_solver_run(benchmark, solver_class):
    """Hook called in `test_solver_run`."""
    if solver_class.name.lower() == "tsb-timesfm":
        if find_spec("timesfm") is None:
            pytest.xfail(
                "TSB-TimesFM needs the optional timesfm package."
            )


def check_test_dataset_get_data(benchmark, dataset_class):
    if dataset_class.name.lower() in [
        "daphnet", "ecg", "genesis", "ghl",
        "iops", "kdd21", "mgab",
        "occupancy", "opportunity", "sensorscope", "smd",
        "svdb", "yahoo", "nab", "mitdb", "dodgers",
    ]:
        pytest.xfail(f"{dataset_class.name} dataset is not downloaded.")

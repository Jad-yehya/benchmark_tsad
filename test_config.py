import sys  # noqa: F401

import pytest  # noqa: F401

from benchopt.utils.sys_info import get_cuda_version


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if solver_class.name.lower() == "dif":
        if get_cuda_version() is None:
            pytest.xfail("Deep IsolationForest needs a working GPU hardware.")

    if solver_class.name.lower() == "lstm":
        if get_cuda_version() is None:
            pytest.xfail("LSTM needs a working GPU hardware.")

"""Custom metrics to be used by example_metrics.py

They reside in a different file so they can be used by Auto-sklearn."""

import numpy as np


############################################################################
# Custom metrics definition
# =========================

def accuracy(solution, prediction):
    # custom function defining accuracy
    return np.mean(solution == prediction)


def error(solution, prediction):
    # custom function defining error
    return np.mean(solution != prediction)


def accuracy_wk(solution, prediction, dummy):
    # custom function defining accuracy and accepting an additional argument
    assert dummy is None
    return np.mean(solution == prediction)


def error_wk(solution, prediction, dummy):
    # custom function defining error and accepting an additional argument
    assert dummy is None
    return np.mean(solution != prediction)

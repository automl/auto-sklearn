# -*- encoding: utf-8 -*-
# REGRESSION METRICS (work on raw solution and prediction)
# These can be computed on all solutions and predictions
# (classification included)
from __future__ import print_function

import numpy as np

from autosklearn.constants import REGRESSION, METRIC_TO_STRING
from autosklearn.metrics.common import mv_mean


def calculate_score(metric, solution, prediction):
    metric = METRIC_TO_STRING[metric]
    return globals()[metric](solution, prediction)


def r2_metric(solution, prediction, task=REGRESSION):
    """
    1 - Mean squared error divided by variance
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    mse = mv_mean((solution - prediction) ** 2)
    var = mv_mean((solution - mv_mean(solution)) ** 2)
    score = 1 - mse / var
    return mv_mean(score)


def a_metric(solution, prediction, task=REGRESSION):
    """
    1 - Mean absolute error divided by mean absolute deviation
    :param solution:
    :param prediction:
    :param task:
    :return:
    """
    mae = mv_mean(np.abs(solution - prediction))  # mean absolute error
    mad = mv_mean(
        np.abs(solution - mv_mean(solution)))  # mean absolute deviation
    score = 1 - mae / mad
    return mv_mean(score)

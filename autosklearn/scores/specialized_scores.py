# -*- encoding: utf-8 -*-

# ======= Specialized scores ========
# We run all of them for all tasks even though they don't make sense for
# some tasks
from __future__ import print_function

from autosklearn.constants import BINARY_CLASSIFICATION, \
    MULTICLASS_CLASSIFICATION
from autosklearn.scores.classification_metrics import bac_metric, pac_metric, \
    f1_metric


def nbac_binary_score(solution, prediction):
    """
    Normalized balanced accuracy for binary and multilabel
    classification.
    :param solution:
    :param prediction:
    :return:
    """
    return bac_metric(solution, prediction, task=BINARY_CLASSIFICATION)


def nbac_multiclass_score(solution, prediction):
    """
    Multiclass accuracy for binary and multilabel classification.
    :param solution:
    :param prediction:
    :return:
    """
    return bac_metric(solution, prediction, task=MULTICLASS_CLASSIFICATION)


def npac_binary_score(solution, prediction):
    """
    Normalized balanced accuracy for binary and multilabel
    classification.
    :param solution:
    :param prediction:
    :return:
    """
    return pac_metric(solution, prediction, task=BINARY_CLASSIFICATION)


def npac_multiclass_score(solution, prediction):
    """
    Multiclass accuracy for binary and multilabel classification.
    :param solution:
    :param prediction:
    :return:
    """
    return pac_metric(solution, prediction, task=MULTICLASS_CLASSIFICATION)


def f1_binary_score(solution, prediction):
    """
    Normalized balanced accuracy for binary and multilabel
    classification.
    :param solution:
    :param prediction:
    :return:
    """
    return f1_metric(solution, prediction, task=BINARY_CLASSIFICATION)


def f1_multiclass_score(solution, prediction):
    """
    Multiclass accuracy for binary and multilabel classification.
    :param solution:
    :param prediction:
    :return:
    """
    return f1_metric(solution, prediction, task=MULTICLASS_CLASSIFICATION)

# -*- encoding: utf-8 -*-
import numpy as np
from autosklearn.constants import MULTICLASS_CLASSIFICATION, \
    BINARY_CLASSIFICATION


def sanitize_array(array):
    """
    Replace NaN and Inf (there should not be any!)
    :param array:
    :return:
    """
    a = np.ravel(array)
    maxi = np.nanmax(a[np.isfinite(a)])
    mini = np.nanmin(a[np.isfinite(a)])
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    return array


def binarize_predictions(array, task=BINARY_CLASSIFICATION):
    """
    Turn predictions into decisions {0,1} by selecting the class with largest
    score for multi class problems and thresh holding at 0.5 for other cases.

    :param array:
    :param task:
    :return:
    """
    # add a very small random value as tie breaker (a bit bad because
    # this changes the score every time)
    # so to make sure we get the same result every time, we seed it
    # eps = 1e-15
    # np.random.seed(sum(array.shape))
    # array = array + eps*np.random.rand(array.shape[0],array.shape[1])
    bin_array = np.zeros(array.shape, dtype=np.int32)
    if (task != MULTICLASS_CLASSIFICATION) or (array.shape[1] == 1):
        bin_array[array >= 0.5] = 1
    else:
        sample_num = array.shape[0]
        for i in range(sample_num):
            j = np.argmax(array[i, :])
            bin_array[i, j] = 1
    return bin_array


def create_multiclass_solution(solution, prediction):
    solution_binary = np.zeros((prediction.shape), dtype=np.int32)
    for i in range(solution_binary.shape[0]):
        try:
            solution_binary[i, int(solution[i])] = 1
        except IndexError as e:
            raise IndexError('too many indices to array. array has shape %s, '
                             'indices are "%s %s"' %
                             (solution_binary.shape, str(i), solution[i]))
    return solution_binary

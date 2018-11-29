# -*- encoding: utf-8 -*-
# Functions performing various data conversions for the ChaLearn AutoML
# challenge

import numpy as np

__all__ = [
    'predict_RAM_usage',
    'convert_to_num',
    'convert_to_bin'
]


def binarization(array):
    # Takes a binary-class datafile and turn the max value (positive class)
    # into 1 and the min into 0
    array = np.array(array, dtype=float)  # conversion needed to use np.inf
    if len(np.unique(array)) > 2:
        raise ValueError('The argument must be a binary-class datafile. '
                         '{} classes detected'.format(len(np.unique(array))))

    # manipulation which aims at avoid error in data
    # with for example classes '1' and '2'.
    array[array == np.amax(array)] = np.inf
    array[array == np.amin(array)] = 0
    array[array == np.inf] = 1
    return np.array(array, dtype=int)


def multilabel_to_multiclass(array):
    array = binarization(array)
    return np.array([np.nonzero(array[i, :])[0][0] for i in range(len(array))])


def convert_to_num(Ybin):
    """
    Convert binary targets to numeric vector
    typically classification target values
    :param Ybin:
    :return:
    """
    result = np.array(Ybin)
    if len(Ybin.shape) != 1:
        result = np.dot(Ybin, range(Ybin.shape[1]))
    return result


def convert_to_bin(Ycont, nval, verbose=True):
    # Convert numeric vector to binary (typically classification target values)
    if verbose:
        pass
    Ybin = [[0] * nval for x in range(len(Ycont))]
    for i in range(len(Ybin)):
        line = Ybin[i]
        line[np.int(Ycont[i])] = 1
        Ybin[i] = line
    return Ybin


def predict_RAM_usage(X, categorical):
    # Return estimated RAM usage of dataset after OneHotEncoding in bytes.
    estimated_columns = 0
    for i, cat in enumerate(categorical):
        if cat:
            unique_values = np.unique(X[:, i])
            num_unique_values = np.sum(np.isfinite(unique_values))
            estimated_columns += num_unique_values
        else:
            estimated_columns += 1
    estimated_ram = estimated_columns * X.shape[0] * X.dtype.itemsize
    return estimated_ram

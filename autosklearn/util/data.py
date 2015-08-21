# -*- encoding: utf-8 -*-
# Functions performing various data conversions for the ChaLearn AutoML
# challenge

from __future__ import print_function

__all__ = [
    'predict_RAM_usage',
    'save_predictions',
    'convert_to_num',
]
import glob
import os

import numpy as np


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
    Ybin = [[0] * nval for x in xrange(len(Ycont))]
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


# ================ Output prediction results and prepare code submission =
def save_predictions(filename, predictions):
    # Write prediction scores in prescribed format
    with open(filename, 'w') as output_file:
        for row in predictions:
            if not isinstance(row, np.ndarray) and not isinstance(row, list):
                row = [row]
            for val in row:
                output_file.write('{:g} '.format(float(val)))
            output_file.write('\n')


# ================ Inventory input data and create data structure ========
def inventory_data(input_dir):
    # Inventory the datasets in the input directory and return them in
    # alphabetical order

    # Assume first that there is a hierarchy dataname/dataname_train.data
    training_names = _inventory_data_dir(input_dir)
    ntr = len(training_names)
    if ntr == 0:
        # Try to see if there is a flat directory structure
        training_names = _inventory_data_nodir(input_dir)
    ntr = len(training_names)
    if ntr == 0:
        print('WARNING: Inventory data - No data file found')
        training_names = []
    training_names.sort()
    return training_names


def _inventory_data_nodir(input_dir):
    # Inventory data, assuming flat directory structure
    training_names = glob.glob(os.path.join(input_dir, '*_train.data'))
    for i in range(0, len(training_names)):
        name = training_names[i]
        training_names[
            i] = name[-name[::-1].index(os.sep):-name[::-1].index('_') - 1]
        _check_dataset(input_dir, training_names[i])
    return training_names


def _inventory_data_dir(input_dir):
    # Inventory data, assuming flat directory structure,
    # assuming a directory hierarchy.

    # This supports subdirectory structures obtained by concatenating bundles
    training_names = glob.glob(input_dir + '/*/*_train.data')
    for i in range(0, len(training_names)):
        name = training_names[i]
        training_names[
            i] = name[-name[::-1].index(os.sep):-name[::-1].index('_') - 1]
        _check_dataset(os.path.join(input_dir, training_names[i]),
                       training_names[i])
    return training_names


def _check_dataset(dirname, name):
    # Check the test and valid files are in the directory,
    # as well as the solution
    valid_file = os.path.join(dirname, name + '_valid.data')
    if not os.path.isfile(valid_file):
        print('No validation file for ' + name)
        exit(1)
    test_file = os.path.join(dirname, name + '_test.data')
    if not os.path.isfile(test_file):
        print('No test file for ' + name)
        exit(1)
    # Check the training labels are there
    training_solution = os.path.join(dirname, name + '_train.solution')
    if not os.path.isfile(training_solution):
        print('No training labels for ' + name)
        exit(1)
    return True

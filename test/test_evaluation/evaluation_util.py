import functools
import os
import sys
import traceback
import unittest

import numpy as np
from numpy.linalg import LinAlgError
import sklearn.datasets

from autosklearn.constants import *
from autosklearn.util.data import convert_to_bin
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.pipeline.util import get_dataset

N_TEST_RUNS = 5


class Dummy(object):
    def __init__(self):
        self.name = 'Dummy'


class BaseEvaluatorTest(unittest.TestCase):
    def __init__(self, methodName):
        super(BaseEvaluatorTest, self).__init__(methodName)
        self.output_directories = []

    def _fit(self, evaluator):
        return self.__fit(evaluator.fit)

    def _partial_fit(self, evaluator, fold):
        partial_fit = functools.partial(evaluator.partial_fit, fold=fold)
        return self.__fit(partial_fit)

    def __fit(self, function_handle):
        """Allow us to catch known and valid exceptions for all evaluate
        scripts."""
        try:
            function_handle()
            return True
        except KeyError as e:
            if 'Floating-point under-/overflow occurred at epoch' in \
                    e.args[0] or \
                    'removed all features' in e.args[0] or \
                    'failed to create intent' in e.args[0]:
                pass
            else:
                traceback.print_exc()
                raise e
        except ValueError as e:
            if 'Floating-point under-/overflow occurred at epoch' in e.args[
                0] or \
                            'removed all features' in e.args[0] or \
                            'failed to create intent' in e.args[0]:
                pass
            else:
                raise e
        except LinAlgError as e:
            if 'not positive definite, even with jitter' in e.args[0]:
                pass
            else:
                raise e
        except RuntimeWarning as e:
            if 'invalid value encountered in sqrt' in e.args[0]:
                pass
            elif 'divide by zero encountered in divide' in e.args[0]:
                pass
            else:
                raise e
        except UserWarning as e:
            if 'FastICA did not converge' in e.args[0]:
                pass
            else:
                raise e


def get_multiclass_classification_datamanager():
    X_train, Y_train, X_test, Y_test = get_dataset('iris')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_valid = X_test[:25, ]
    Y_valid = Y_test[:25, ]
    X_test = X_test[25:, ]
    Y_test = Y_test[25:, ]

    D = Dummy()
    D.info = {
        'task': MULTICLASS_CLASSIFICATION,
        'is_sparse': False,
        'label_num': 3
    }
    D.data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'X_test': X_test,
        'Y_test': Y_test
    }
    D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']
    return D


def get_abalone_datamanager():
    dataset = 'abalone'
    dataset_path = os.path.join(os.path.dirname(__file__), '.datasets',
                                dataset)
    D = CompetitionDataManager(dataset_path)
    return D


def get_multilabel_classification_datamanager():
    X_train, Y_train, X_test, Y_test = get_dataset('iris')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    Y_train = np.array(convert_to_bin(Y_train, 3))
    #for i in range(Y_train_.shape[0]):
    #    Y_train_[:, Y_train[i]] = 1
    #Y_train = Y_train_
    Y_test = np.array(convert_to_bin(Y_test, 3))
    #for i in range(Y_test_.shape[0]):
    #    Y_test_[:, Y_test[i]] = 1
    #Y_test = Y_test_

    X_valid = X_test[:25, ]
    Y_valid = Y_test[:25, ]
    X_test = X_test[25:, ]
    Y_test = Y_test[25:, ]

    D = Dummy()
    D.info = {
        'task': MULTILABEL_CLASSIFICATION,
        'is_sparse': False,
        'label_num': 3
    }
    D.data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_valid': X_valid,
        'Y_valid': Y_valid,
        'X_test': X_test,
        'Y_test': Y_test
    }
    D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']
    return D


def get_binary_classification_datamanager():
    X_train, Y_train, X_test, Y_test = get_dataset('iris')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    eliminate_class_two = Y_train != 2
    X_train = X_train[eliminate_class_two]
    Y_train = Y_train[eliminate_class_two]

    eliminate_class_two = Y_test != 2
    X_test = X_test[eliminate_class_two]
    Y_test = Y_test[eliminate_class_two]

    X_valid = X_test[:25, ]
    Y_valid = Y_test[:25, ]
    X_test = X_test[25:, ]
    Y_test = Y_test[25:, ]

    D = Dummy()
    D.info = {
        'task': BINARY_CLASSIFICATION,
        'is_sparse': False,
        'label_num': 2
    }
    D.data = {
        'X_train': X_train,
        'Y_train': Y_train.reshape((-1, 1)),
        'X_valid': X_valid,
        'Y_valid': Y_valid.reshape((-1, 1)),
        'X_test': X_test,
        'Y_test': Y_test.reshape((-1, 1))
    }
    D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical']
    return D


def get_regression_datamanager():
    X_train, Y_train, X_test, Y_test = get_dataset('boston')
    indices = list(range(X_train.shape[0]))
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_valid = X_test[:200, ]
    Y_valid = Y_test[:200, ]
    X_test = X_test[200:, ]
    Y_test = Y_test[200:, ]

    D = Dummy()
    D.info = {
        'task': REGRESSION,
        'is_sparse': False,
        'label_num': 1
    }
    D.data = {
        'X_train': X_train,
        'Y_train': Y_train.reshape((-1, 1)),
        'X_valid': X_valid,
        'Y_valid': Y_valid.reshape((-1, 1)),
        'X_test': X_test,
        'Y_test': Y_test.reshape((-1, 1))
    }
    D.feat_type = ['numerical', 'Numerical', 'numerical', 'numerical',
                   'numerical', 'numerical', 'numerical', 'numerical',
                   'numerical', 'numerical', 'numerical']
    return D


def get_500_classes_datamanager():
    weights = ([0.002] * 475) + ([0.001] * 25)
    X, Y = sklearn.datasets.make_classification(n_samples=1000,
                                                n_features=20,
                                                n_classes=500,
                                                n_clusters_per_class=1,
                                                n_informative=15,
                                                n_redundant=5,
                                                n_repeated=0,
                                                weights=weights,
                                                flip_y=0,
                                                class_sep=1.0,
                                                hypercube=True,
                                                shift=None,
                                                scale=1.0,
                                                shuffle=True,
                                                random_state=1)

    D = Dummy()
    D.info = {
        'task': MULTICLASS_CLASSIFICATION,
        'is_sparse': False,
        'label_num': 500
    }
    D.data = {'X_train': X[:700], 'Y_train': Y[:700],
              'X_valid': X[700:710], 'Y_valid': Y[700:710],
              'X_test': X[710:], 'Y_test': Y[710:]
              }
    D.feat_type = ['numerical'] * 20
    return D


def get_dataset_getters():
    return [get_binary_classification_datamanager,
            get_multiclass_classification_datamanager,
            get_multilabel_classification_datamanager,
            get_500_classes_datamanager,
            get_abalone_datamanager,
            get_regression_datamanager]

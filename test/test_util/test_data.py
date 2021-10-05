from typing import cast
import warnings

import pytest

import numpy as np
import pandas as pd
import sklearn.datasets
from scipy.sparse import csr_matrix

from autosklearn.constants import (
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
    REGRESSION, MULTIOUTPUT_REGRESSION, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from autosklearn.util.data import reduce_dataset_size_if_too_large


parametrize = pytest.mark.parametrize


@parametrize("task, y", [
    (BINARY_CLASSIFICATION, np.asarray(
        9999 * [0] + 1 * [1]
    )),
    (MULTICLASS_CLASSIFICATION, np.asarray(
        4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]
    )),
    (MULTILABEL_CLASSIFICATION, np.asarray(
        4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]]
    ))
])
def test_subsample_classification_unique_labels_stay_in_training_set(task, y):
    n_samples = 10000
    X = np.random.random(size=(n_samples, 3))
    memory_limit = 1  # Force subsampling

    # Make sure our test assumptions are correct
    assert len(y) == n_samples, "Ensure tests are correctly setup"

    values, counts = np.unique(y, axis=0, return_counts=True)
    unique_labels = [value for value, count in zip(values, counts) if count == 1]
    assert len(unique_labels), "Ensure we have unique labels in the test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, y_sampled = reduce_dataset_size_if_too_large(
            X, y,
            seed=1,
            memory_limit=memory_limit,
            is_classification=task in CLASSIFICATION_TASKS
        )

    assert len(y_sampled) <= len(y)
    assert all(label in y_sampled for label in unique_labels), \
        f"sampled unique = {np.unique(y_sampled)}, original unique = {unique_labels}"


@parametrize('memory_limit', [1, 100, None])
@parametrize('precision', [float, np.float32, np.float64, np.float128])
@parametrize('task', CLASSIFICATION_TASKS + REGRESSION_TASKS)
@parametrize('Xtype', ['list', 'dataframe', 'ndarray', 'sparse'])
def test_reduce_dataset_size_if_too_large(memory_limit, precision, task, Xtype):
    fixture = {
        BINARY_CLASSIFICATION: {
            1: {float: 2500, np.float32: 2500, np.float64: 2500, np.float128: 1250},
            100: {float: 12000, np.float32: 12000, np.float64: 12000, np.float128: 12000},
            None: {float: 12000, np.float32: 12000, np.float64: 12000, np.float128: 12000},
        },
        MULTICLASS_CLASSIFICATION: {
            1: {float: 390, np.float32: 390, np.float64: 390, np.float128: 195},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
            None: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        MULTILABEL_CLASSIFICATION: {
            1: {float: 390, np.float32: 390, np.float64: 390, np.float128: 195},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
            None: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        REGRESSION: {
            1: {float: 1250, np.float32: 1250, np.float64: 1250, np.float128: 625},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
            None: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        },
        MULTIOUTPUT_REGRESSION: {
            1: {float: 1250, np.float32: 1250, np.float64: 1250, np.float128: 625},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
            None: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        }
    }

    # Generate data
    if task == BINARY_CLASSIFICATION:
        X, y = sklearn.datasets.make_hastie_10_2()

    elif task == MULTICLASS_CLASSIFICATION:
        X, y = sklearn.datasets.load_digits(return_X_y=True)

    elif task == MULTILABEL_CLASSIFICATION:
        X, y_ = sklearn.datasets.load_digits(return_X_y=True)
        X, y_ = cast(np.ndarray, X), cast(np.ndarray, y_)
        y = np.zeros((X.shape[0], 10))
        for i, j in enumerate(y_):
            y[i, j] = 1

    elif task == REGRESSION:
        X, y = sklearn.datasets.make_friedman1(n_samples=5000, n_features=20)

    elif task == MULTIOUTPUT_REGRESSION:
        X, y = sklearn.datasets.make_friedman1(n_samples=5000, n_features=20)
        y = np.vstack((y, y)).transpose()

    else:
        raise ValueError(task)

    # Cast to np.ndarray and ensure sizes fit
    X, y = cast(np.ndarray, X), cast(np.ndarray, y)
    assert X.shape[0] == y.shape[0]

    # Convert to precision and store it's dtype
    X = X.astype(precision)
    old_dtype = X.dtype

    # Convert to the specified task type
    if Xtype == 'list':
        X = X.tolist()
    elif Xtype == 'dataframe':
        X = pd.DataFrame(X)
    elif Xtype == 'spmatrix':
        X = csr_matrix(X)
        y = csr_matrix(y)
    else:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_new, _ = reduce_dataset_size_if_too_large(
            X, y,
            seed=1,
            memory_limit=memory_limit,
            is_classification=task in CLASSIFICATION_TASKS
        )

    # Assert there was subsampling when expected
    expected_n_samples = fixture[task][memory_limit][precision]
    assert X_new.shape[0] == expected_n_samples

    # Assert precision reduction when we have a memory limit of 1MB
    if memory_limit == 1:
        expected_dtypes = {
            np.float128: np.float64,
            np.float64: np.float32,
            np.float32: np.float32,
            float: np.float32
        }
        assert X_new.dtype == expected_dtypes[precision]
    elif Xtype == 'list':
        if precision == np.float128:
            #   ndarray[np.float128] -> List[np.float128]
            #   List[np.float128] -> ndarray[np.float128]
            assert X_new.dtype == old_dtype
        else:
            #   ndarray[np.float{64,32}] -> List[float]
            #   List[float] -> ndarray[np.float64]
            assert X_new.dtype == np.float64
    else:
        assert X_new.dtype == old_dtype

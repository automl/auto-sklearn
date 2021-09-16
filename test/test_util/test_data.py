import itertools
import warnings

import pytest
import numpy as np
import sklearn.datasets

from autosklearn.constants import (
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
    REGRESSION, MULTIOUTPUT_REGRESSION, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from autosklearn.util.data import reduce_dataset_size_if_too_large


@pytest.mark.parametrize("task, y", [
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


@pytest.mark.parametrize(
    'memory_limit,precision,task',
    [
        (memory_limit, precision, task)
        for task in itertools.chain(CLASSIFICATION_TASKS, REGRESSION_TASKS)
        for precision in (float, np.float32, np.float64, np.float128)
        for memory_limit in (1, 100, None)
    ]
)
def test_reduce_dataset_size_if_too_large(memory_limit, precision, task):
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
    if task == BINARY_CLASSIFICATION:
        X, y = sklearn.datasets.make_hastie_10_2()
    elif task == MULTICLASS_CLASSIFICATION:
        X, y = sklearn.datasets.load_digits(return_X_y=True)
    elif task == MULTILABEL_CLASSIFICATION:
        X, y_ = sklearn.datasets.load_digits(return_X_y=True)
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

    X = X.astype(precision)
    old_dtype = X.dtype
    assert X.shape[0] == y.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_new, y_new = reduce_dataset_size_if_too_large(
            X, y,
            seed=1,
            memory_limit=memory_limit,
            is_classification=task in CLASSIFICATION_TASKS
        )

    # Assert there was subsampling when expected
    assert X_new.shape[0] == fixture[task][memory_limit][precision]

    # Assert precision reduction whewn we have a memory limit of 1MB
    if memory_limit == 1:
        expected_dtypes = {
            np.float128: np.float64,
            np.float64: np.float32,
            np.float32: np.float32,
            float: np.float32
        }
        assert X_new.dtype == expected_dtypes[precision]
    else:
        assert old_dtype == X_new.dtype

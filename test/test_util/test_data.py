from typing import List, Dict
from itertools import chain
import warnings

import pytest

import numpy as np
import pandas as pd
import sklearn.datasets
from scipy.sparse import csr_matrix, spmatrix

from autosklearn.constants import (
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
    REGRESSION, MULTIOUTPUT_REGRESSION, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from autosklearn.util.data import (
    subsample,
    reduce_dataset_size_if_too_large,
    reduce_precision,
    reduction_mapping,
    supported_precision_reductions,
)

parametrize = pytest.mark.parametrize


@parametrize("y", [
    np.asarray(9999 * [0] + 1 * [1]),
    np.asarray(4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]),
    np.asarray(4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]])
])
@parametrize("random_state", list(range(5)))
def test_subsample_classification_unique_labels_stay_in_training_set(y, random_state):
    n_samples = len(y)
    X = np.random.random(size=(n_samples, 3))
    sample_size = 100

    values, counts = np.unique(y, axis=0, return_counts=True)
    unique_labels = [value for value, count in zip(values, counts) if count == 1]
    assert len(unique_labels), "Ensure we have unique labels in the test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_sampled, y_sampled = subsample(
            X, y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=True
        )

    assert X_sampled.dtype == X.dtype and y_sampled.dtype == y.dtype
    assert len(y_sampled) == sample_size
    assert all(label in y_sampled for label in unique_labels), \
        f"sampled unique = {np.unique(y_sampled)}, original unique = {unique_labels}"


@parametrize("X", [np.asarray([[1, 1, 1]] * 30)])
@parametrize("x_type", [list, np.ndarray, csr_matrix, pd.DataFrame])
@parametrize("y, task", [
    (np.asarray([0] * 15 + [1] * 15), BINARY_CLASSIFICATION),
    (np.asarray([0] * 10 + [1] * 10 + [2] * 10), MULTICLASS_CLASSIFICATION),
    (np.asarray([[1, 0, 1]] * 30), MULTILABEL_CLASSIFICATION),
    (np.asarray([1.0] * 30), REGRESSION),
    (np.asarray([[1.0, 1.0, 1.0]] * 30), MULTIOUTPUT_REGRESSION),
])
@parametrize("y_type", [list, np.ndarray, pd.DataFrame, pd.Series])
@parametrize("random_state", [0])
@parametrize("sample_size", [0.25, 0.5, 5, 10])
def test_subsample_validity(X, x_type, y, y_type, random_state, sample_size, task):
    """ Asserts the validity of the function with all valid types

    We want to make sure that `subsample` works correctly with all the types listed
    as x_type and y_type.

    We also want to make sure it works with all kinds of target types.

    The output should maintain the types, and subsample the correct amount.
    """
    assert len(X) == len(y)  # Make sure our test data is correct

    if (
        y_type == pd.Series
        and task in [MULTILABEL_CLASSIFICATION, MULTIOUTPUT_REGRESSION]
    ):
        # We can't have a pd.Series with multiple values as it's 1 dimensional
        pytest.skip()

    # Convert our data to its given x_type or y_type
    def convert(arr, objtype):
        if objtype == np.ndarray:
            return arr
        elif objtype == list:
            return arr.tolist()
        else:
            return objtype(arr)

    X = convert(X, x_type)
    y = convert(y, y_type)

    # Subsample the data, ignoring any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_sampled, y_sampled = subsample(
            X, y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=task in CLASSIFICATION_TASKS
        )

    # Function to get the type of an obj
    def dtype(obj):

        if isinstance(obj, List):
            if isinstance(obj[0], List):
                return type(obj[0][0])
            else:
                return type(obj[0])

        elif isinstance(obj, pd.DataFrame):
            return obj.dtypes

        else:
            return obj.dtype

    # Check that the types of X remain the same after subsampling
    if isinstance(X, pd.DataFrame):
        # Dataframe can have multiple types, one per column
        assert list(dtype(X_sampled)) == list(dtype(X))
    else:
        assert dtype(X_sampled) == dtype(X)

    # Check that the types of y remain the same after subsampling
    if isinstance(y, pd.DataFrame):
        assert list(dtype(y_sampled)) == list(dtype(y))
    else:
        assert dtype(y_sampled) == dtype(y)

    # Function to get the size of an object
    def size(obj):
        if isinstance(obj, spmatrix):  # spmatrix doesn't support __len__
            return obj.shape[0] if obj.shape[0] > 1 else obj.shape[1]
        else:
            return len(obj)

    # check the right amount of samples were taken
    if sample_size < 1:
        assert size(X_sampled) == int(sample_size * size(X))
    else:
        assert size(X_sampled) == sample_size


@parametrize('X', [np.asarray([[0, 0, 1]] * 10)])
@parametrize('dtype', supported_precision_reductions + [np.dtype('float32'), np.dtype('float64')])
@parametrize('x_type', [np.ndarray, csr_matrix])
def test_reduce_precision_correctly_reduces_precision(X, dtype, x_type):
    X = X.astype(dtype)
    if x_type == csr_matrix:
        X = x_type(X)

    X_reduced, precision = reduce_precision(X)

    # Check the reduced precision is correctly returned
    assert X_reduced.dtype == precision

    # Check that it was reduce to the correct precision
    expected: Dict[type, type] = {
        np.float32: np.float32,
        np.float64: np.float32,
        np.dtype('float32'): np.float32,
        np.dtype('float64'): np.float32
    }
    if hasattr(np, 'float96'):
        expected[np.float96] = np.float64

    if hasattr(np, 'float128'):
        expected[np.float128] = np.float64

    assert precision == expected[dtype]

    # Check that X's shape was not modified in any way
    assert X.shape == X_reduced.shape

    # Check that the return type is the one as we passed in
    assert type(X) == type(X_reduced)


@parametrize('X', [np.asarray([0, 0, 1]) * 10])
@parametrize('dtype', [np.int32, np.int64, np.complex128])
def test_reduce_precision_with_unsupported_dtypes(X, dtype):
    X = X.astype(dtype)
    with pytest.raises(ValueError) as err:
        reduce_precision(X)

    expected = f"X.dtype = {X.dtype} not equal to any supported {supported_precision_reductions}"
    assert err.value.args[0] == expected


@parametrize("X", [
    np.asarray([[1] * 10] * 100000, dtype=np.float64)  # Make it big for reductions to take place
])
@parametrize("x_type", [csr_matrix, np.ndarray])
@parametrize("dtype", supported_precision_reductions)
@parametrize('y, is_classification', [
    (np.asarray([1] * 100000), True),
    (np.asarray([1.0] * 100000), False),
])
@parametrize('multiplier', [1, 5.2, 10])
@parametrize('operations', [['precision'], ['subsample'], ['precision', 'subsample']])
def test_reduce_dataset_reduces_size_and_precision(
    X, x_type, dtype, y, is_classification, multiplier, operations
):
    assert len(X) == len(y)
    X = X.astype(dtype)
    if x_type == csr_matrix:
        X = x_type(X)

    random_state = 0
    memory_limit = 1  # Force reductions

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        X_out, y_out = reduce_dataset_size_if_too_large(
            X=X,
            y=y,
            random_state=random_state,
            memory_limit=memory_limit,
            operations=operations,
            multiplier=multiplier,
            is_classification=is_classification,
        )

    def bytes(arr):
        return arr.nbytes if isinstance(arr, np.ndarray) else arr.data.nbytes

    # If we expect some precision reduction unless at float32 already
    if 'precision' in operations and dtype != np.float32:
        expected = reduction_mapping[X.dtype]
        assert X_out.dtype == expected
        assert bytes(X_out) < bytes(X)

    # If we expect some subsampling
    if 'subsample' in operations:
        assert X_out.shape[0] < X.shape[0]
        assert y_out.shape[0] < y.shape[0]
        assert bytes(X_out) < bytes(X)


def test_reduce_dataset_invalid_dtype_for_precision_reduction():
    dtype = int
    X = np.asarray([1, 2, 3], dtype=dtype)

    with pytest.raises(ValueError) as err:
        reduce_dataset_size_if_too_large(
            X=X,
            y=X,
            operations=['precision'],
            memory_limit=1,
            is_classification=False
        )

    expected_err = f"Unsupported type `{X.dtype}` for precision reduction"
    assert err.value.args[0] == expected_err


def test_reduce_dataset_invalid_operations():
    invalid_op = "invalid"

    X = np.asarray([1, 2, 3], dtype=float)
    with pytest.raises(ValueError) as err:
        reduce_dataset_size_if_too_large(
            X=X,
            y=X,
            operations=[invalid_op],
            memory_limit=1,
            is_classification=False
        )

    expected_err = f"Unknown operation `{invalid_op}`"
    assert err.value.args[0] == expected_err


@pytest.mark.parametrize(
    'memory_limit,precision,task',
    [
        (memory_limit, precision, task)
        for task in chain(CLASSIFICATION_TASKS, REGRESSION_TASKS)
        for precision in (float, np.float32, np.float64, np.float128)
        for memory_limit in (1, 100)
    ]
)
def test_reduce_dataset_subsampling_explicit_values(memory_limit, precision, task):
    random_state = 0
    fixture = {
        BINARY_CLASSIFICATION: {
            1: {float: 2500, np.float32: 2500, np.float64: 2500, np.float128: 1250},
            100: {float: 12000, np.float32: 12000, np.float64: 12000, np.float128: 12000},
        },
        MULTICLASS_CLASSIFICATION: {
            1: {float: 390, np.float32: 390, np.float64: 390, np.float128: 195},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        MULTILABEL_CLASSIFICATION: {
            1: {float: 390, np.float32: 390, np.float64: 390, np.float128: 195},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        REGRESSION: {
            1: {float: 1250, np.float32: 1250, np.float64: 1250, np.float128: 625},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        },
        MULTIOUTPUT_REGRESSION: {
            1: {float: 1250, np.float32: 1250, np.float64: 1250, np.float128: 625},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        }
    }

    # Create the task and data
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

    # Validate the test data and make sure X and y have the same number of rows
    assert X.shape[0] == y.shape[0]

    # Convert X to the dtype we are testing
    X = X.astype(precision)

    # Preform the subsampling through `reduce_dataset_size_if_too_large`
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_new, y_new = reduce_dataset_size_if_too_large(
            X=X, y=y,
            random_state=random_state,
            memory_limit=memory_limit,
            is_classification=task in CLASSIFICATION_TASKS,
            operations=['precision', 'subsample'],
            multiplier=10
        )

    # Assert the new number of samples
    assert X_new.shape[0] == fixture[task][memory_limit][precision]

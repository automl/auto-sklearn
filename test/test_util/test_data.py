import warnings

import pytest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, spmatrix

from autosklearn.constants import (
    BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
    REGRESSION, MULTIOUTPUT_REGRESSION, CLASSIFICATION_TASKS
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
    """ Asserts the validity of the function with all valid types """
    assert len(X) == len(y)

    if (
        y_type == pd.Series
        and task in [MULTILABEL_CLASSIFICATION, MULTIOUTPUT_REGRESSION]
    ):
        pytest.skip()

    def convert(arr, objtype):
        if objtype == np.ndarray:
            return arr
        elif objtype == list:
            return arr.tolist()
        else:
            return objtype(arr)

    X = convert(X, x_type)
    y = convert(y, y_type)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_sampled, y_sampled = subsample(
            X, y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=task in CLASSIFICATION_TASKS
        )

    # Check the types remain the same
    def dtype(obj):
        if isinstance(obj, list):
            if isinstance(obj[0], list):
                return type(obj[0][0])
            else:
                return type(obj[0])
        elif isinstance(obj, pd.DataFrame):
            return obj.dtypes
        else:
            return obj.dtype

    if isinstance(X, pd.DataFrame):
        assert list(dtype(X_sampled)) == list(dtype(X))
    else:
        print(X)
        assert dtype(X_sampled) == dtype(X)

    if isinstance(y, pd.DataFrame):
        assert list(dtype(y_sampled)) == list(dtype(y))
    else:
        assert dtype(y_sampled) == dtype(y)

    # Check the right amount of samples were taken
    def size(obj):
        if isinstance(obj, spmatrix):
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
    assert precision == reduction_mapping[dtype]

    # Check that X's shape was not modified in any way
    assert X.shape == X_reduced.shape

    # Check that the return type is the one as we passed in
    assert type(X) == type(X_reduced)


@parametrize('X', [np.asarray([0, 0, 1]) * 10])
@parametrize('dtype', [np.int32, np.int64, np.complex128])
def test_reduce_precision_with_unsupported_dtypes(X, dtype):
    X = X.astype(dtype)
    with pytest.raises(ValueError):
        reduce_precision(X)


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
    X = np.asarray([1, 2, 3], dtype=int)
    with pytest.raises(ValueError):
        reduce_dataset_size_if_too_large(
            X=X,
            y=X,
            operations=['precision'],
            memory_limit=1,
            is_classification=False
        )


def test_reduce_dataset_invalid_operations():
    X = np.asarray([1, 2, 3], dtype=int)
    with pytest.raises(ValueError):
        reduce_dataset_size_if_too_large(
            X=X,
            y=X,
            operations=['invalid'],
            memory_limit=1,
            is_classification=False
        )

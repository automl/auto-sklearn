from typing import List, Tuple, cast
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
    byte_size,
    subsample,
    reduce_dataset_size_if_too_large,
    dtype_to_byte_mapping,
    reduction_mapping
)

parametrize = pytest.mark.parametrize


def test_supported_dtypes_in_reduction_mapping():
    assert set(dtype_to_byte_mapping.keys()) == set(reduction_mapping.keys())


@parametrize('dtype, expected', list(dtype_to_byte_mapping.items()))
def test_byte_size_converts_valid_dtypes(dtype, expected):
    assert byte_size(dtype) == expected


@parametrize('dtype', [np.int32, np.int64, np.complex128])
def test_byte_size_throws_invalid_dtypes(dtype):
    with pytest.raises(ValueError):
        byte_size(dtype)

@parametrize("y", [
    np.asarray(9999 * [0] + 1 * [1]),
    np.asarray(4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]),
    np.asarray(4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]])
])
@parametrize("random_state", list(range(5)))
def test_subsample_classification_unique_labels_stay_in_training_set(y, random_state):
    n_samples = len(y)
    sample_size = 100
    X = np.random.random(size=(n_samples, 3))

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

    assert X_sampled.dtype == X.dtype and y.dtype == y.dtype
    assert len(y_sampled) == sample_size
    assert all(label in y_sampled for label in unique_labels), \
        f"sampled unique = {np.unique(y_sampled)}, original unique = {unique_labels}"

@parametrize("X", [np.asarray([[1, 1, 1]] * 30)])
@parametrize("x_type", [list, np.ndarray, pd.DataFrame])
@parametrize("y, task", [
    (np.asarray([0] * 15 + [1] * 15), BINARY_CLASSIFICATION),
    (np.asarray([0] * 10 + [1] * 10 + [2] * 10), BINARY_CLASSIFICATION),
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
            return type(obj[0])
        elif isinstance(obj, pd.DataFrame):
            return obj.dtypes
        else:
            return obj.dtype

    if isinstance(X, pd.DataFrame):
        assert list(dtype(X_sampled)) == list(dtype(X))
    else:
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

@parametrize('memory_limit', [1, 100, None])
@parametrize('precision', [float, np.float32, np.float64, np.float128])
@parametrize('task', CLASSIFICATION_TASKS + REGRESSION_TASKS)
@parametrize('Xtype', ['list', 'dataframe', 'ndarray', 'sparse'])
def test_reduce_dataset_size_if_too_large_reduces(memory_limit, precision, task, Xtype):
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


def case_X_feat_types_pandas_dataframe_mixed_types() -> Tuple[pd.DataFrame, None]:
    df = pd.DataFrame({
        'col1': [1, 2, 3],  # int64
        'col2': np.asarray([1.0, 2.0, 3.0], dtype=np.float32),  # float32
        'col3': [1.0, 2.0, 3.0],  # float64
        'col4': np.asarray([1.0, 2.0, 3.0], dtype=np.float128),  # float128
        'col5': pd.Series(['cow', 'cat', 'dog'], dtype='category'),  # category
    })
    feat_types = None
    return df, feat_types


def case_X_feattypes_ndarray_mixed() -> Tuple[np.ndarray, List[str]]:
    x = np.asarray([
        np.asarray([1, 2, 3]),  # dtype int64
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),  # float32
        np.asarray([1.0, 2.0, 3.0]),  # float64
        np.asarray([1.0, 2.0, 3.0], dtype=np.float128),  # float128
        np.asarray(['cow', 'cat', 'dog']),  # <U3 (unicode 3)
    ], dtype=object)  # <- keeps from converting everything to string
    feat_types = ['numerical'] * 4 + ['categorical']
    return x, feat_types


def case_sparse_x_ndarray_int() -> Tuple[csr_matrix, List[str]]:
    """ We assume this is going to always be numerical """
    x = csr_matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    feat_types = ['Numerical'] * 5
    return x, feat_types


@parametrize("X, feat_type", [
    case_X_feat_types_pandas_dataframe_mixed_types(),
    case_X_feattypes_ndarray_mixed(),
    case_sparse_x_ndarray_int(),
])
@parametrize('y, is_classification', [
    (pd.Series([1, 1, 1, 1, 1]), True),
    ([1, 1, 1, 1, 1], True),
    (np.asarray([1, 1, 1, 1, 1]), True),
    (csr_matrix([1, 1, 1, 1, 1]), True),
    (pd.Series([1.0, 1.0, 1.0, 1.0, 1.0]), False),
    ([1.0, 1.0, 1.0, 1.0, 1.0], False),
    (np.asarray([1.0, 1.0, 1.0, 1.0, 1.0]), False),
])
def test_reduce_dataset_size_handles_supported_feat_types_without_error(
    X, y, feat_type, is_classification
):
    """ Should complete without error and return the same types as passed in.

    We make an exception for list as this will be converted anyways and we save
    time from converting back.
    """
    # feat_type ignore for now, use if allowing for more flexible subsampling
    seed = 0
    memory_limit = 1  # Force reductions

    X_out, y_out = reduce_dataset_size_if_too_large(
        X=X,
        y=y,
        random_state=seed,
        memory_limit=memory_limit,
        is_classification=is_classification,
    )

    # Assert it comes out as the correct type
    type_mapping = {
        'list': np.ndarray,
        np.ndarray: np.ndarray,
        pd.DataFrame: pd.DataFrame,
        spmatrix: spmatrix
    }

    expected_X_type = type_mapping[type(X)]
    assert isinstance(X_out, expected_X_type)

    expected_y_type = type_mapping[type(y)]
    assert isinstance(y_out, expected_y_type)

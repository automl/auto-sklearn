from typing import Any, Dict, List, Union

import warnings
from itertools import chain

import numpy as np
import pandas as pd
import sklearn.datasets
from scipy.sparse import csr_matrix, spmatrix

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    REGRESSION_TASKS,
)
from autosklearn.util.data import (
    default_dataset_compression_arg,
    reduce_dataset_size_if_too_large,
    reduce_precision,
    reduction_mapping,
    subsample,
    supported_precision_reductions,
    validate_dataset_compression_arg,
)

import pytest

parametrize = pytest.mark.parametrize


@parametrize("arg", [default_dataset_compression_arg])
def test_validate_dataset_compression_arg_returns_original_dict(arg: Dict[str, Any]):
    """
    Parameters
    ----------
    arg: Dict[str, any]

    Expects
    -------
    * Should validate without error
    * Should return a dict identical to the default
    """
    validate_arg = validate_dataset_compression_arg(arg, memory_limit=10)
    assert validate_arg == arg


@parametrize("memory_allocation", [10, 50, 0.1, 0.5])
def test_validate_dataset_compression_arg_returns_with_memory_allocation(
    memory_allocation: Union[float, int],
):
    """
    Parameters
    ----------
    memory_allocation: Union[float, int]
        A valid memory_allocation

    Expects
    -------
    * Should validate without error
    * Should not modify the memory_allocation arg
    * Should populate the default methods
    """
    arg = {"memory_allocation": memory_allocation}
    validate_arg = validate_dataset_compression_arg(arg, memory_limit=100)

    expected_methods = default_dataset_compression_arg["methods"]

    assert validate_arg["memory_allocation"] == memory_allocation
    assert validate_arg["methods"] == expected_methods


@parametrize(
    "methods",
    [
        ["precision"],
        ["precision", "subsample"],
        ["precision", "precision", "subsample"],
    ],
)
def test_validate_dataset_compression_arg_returns_with_same_methods(
    methods: List[str],
):
    """
    Parameters
    ----------
    methods: List[str]
        A valid list of methods

    Expects
    -------
    * Should validate without error
    * Should not modify the methods arg
    * Should populate memory_allocation with the default
    """
    arg = {"methods": methods}
    validate_arg = validate_dataset_compression_arg(arg, memory_limit=10)

    expected_memory_allocation = default_dataset_compression_arg["memory_allocation"]

    assert validate_arg["memory_allocation"] == expected_memory_allocation
    assert validate_arg["methods"] == methods


@parametrize("bad_arg", [[1337], "hello"])
def test_validate_dataset_compression_arg_raises_error_with_bad_arg(bad_arg: Any):
    """
    Parameters
    ----------
    bad_arg: Any
        An arg which is not a Mapping

    Expects
    -------
    * Should raise a ValueError
    """
    with pytest.raises(ValueError, match=r"Unknown type"):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize("key", ["hello", "hi", "hey"])
def test_validate_dataset_compression_arg_raises_error_with_bad_key(key: str):
    """
    Parameters
    ----------
    key: str
        A key which should not be in the parameters

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {**default_dataset_compression_arg, key: 1337}
    with pytest.raises(ValueError, match=r"Unknown key"):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize("memory_allocation", ["hello", {}, [1, 2, 3]])
def test_validate_dataset_compression_arg_raises_error_with_bad_memory_allocation_type(
    memory_allocation: Any,
):
    """
    Parameters
    ----------
    memory_allocation: Any
        A bad type for memory_allocation

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {"memory_allocation": memory_allocation}
    with pytest.raises(
        ValueError, match=r"key 'memory_allocation' must be an `int` or `float`"
    ):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize("memory_allocation", [-0.5, 0.0, 1.0, 1.5])
def test_validate_dataset_compression_arg_raises_error_with_bad_memory_allocation_float(
    memory_allocation: float,
):
    """
    Parameters
    ----------
    memory_allocation: Any
        A bad float value for memory_allocation

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {"memory_allocation": memory_allocation}

    with pytest.raises(
        ValueError, match=r"key 'memory_allocation' if float must be in \(0, 1\)"
    ):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize(
    "memory_allocation, memory_limit", [(0, 10), (10, 10), (-20, 10), (20, 10)]
)
def test_validate_dataset_compression_arg_raises_error_with_bad_memory_allocation_int(
    memory_allocation: int, memory_limit: int
):
    """
    Parameters
    ----------
    memory_allocation: int
        A bad float int for memory_allocation

    memory_limit: int
        The memory limit

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {"memory_allocation": memory_allocation}
    with pytest.raises(
        ValueError, match=r"key 'memory_allocation' if int must be in \(0,"
    ):
        validate_dataset_compression_arg(bad_arg, memory_limit=memory_limit)


@parametrize("methods", [10, {"hello", "world"}, []])
def test_validate_dataset_compression_arg_raises_error_with_bad_methods_type(
    methods: Any,
):
    """
    Parameters
    ----------
    methods: int
        A bad type for key methods

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {"methods": methods}
    with pytest.raises(ValueError, match=r"key 'methods' must be a non-empty list"):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize(
    "methods",
    [
        ["bad", "worse"],
        ["precision", "kind_of_bad"],
        ["still_bad", "precision", "subsample"],
    ],
)
def test_validate_dataset_compression_arg_raises_error_with_bad_methods_entries(
    methods: Any,
):
    """
    Parameters
    ----------
    methods: int
        A bad type for key methods

    Expects
    -------
    * Should raise a ValueError
    """
    bad_arg = {"methods": methods}
    with pytest.raises(ValueError, match=r"key 'methods' can only contain"):
        validate_dataset_compression_arg(bad_arg, memory_limit=10)


@parametrize(
    "y",
    [
        np.asarray(9999 * [0] + 1 * [1]),
        np.asarray(4999 * [1] + 4999 * [2] + 1 * [3] + 1 * [4]),
        np.asarray(
            4999 * [[0, 1, 1]] + 4999 * [[1, 1, 0]] + 1 * [[1, 0, 1]] + 1 * [[0, 0, 0]]
        ),
    ],
)
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
            X,
            y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=True,
        )

    assert X_sampled.dtype == X.dtype and y_sampled.dtype == y.dtype
    assert len(y_sampled) == sample_size
    assert all(
        label in y_sampled for label in unique_labels
    ), f"sampled unique = {np.unique(y_sampled)}, original unique = {unique_labels}"


@parametrize("X", [np.asarray([[1, 1, 1]] * 30)])
@parametrize("x_type", [list, np.ndarray, csr_matrix, pd.DataFrame])
@parametrize(
    "y, task",
    [
        (np.asarray([0] * 15 + [1] * 15), BINARY_CLASSIFICATION),
        (np.asarray([0] * 10 + [1] * 10 + [2] * 10), MULTICLASS_CLASSIFICATION),
        (np.asarray([[1, 0, 1]] * 30), MULTILABEL_CLASSIFICATION),
        (np.asarray([1.0] * 30), REGRESSION),
        (np.asarray([[1.0, 1.0, 1.0]] * 30), MULTIOUTPUT_REGRESSION),
    ],
)
@parametrize("y_type", [list, np.ndarray, pd.DataFrame, pd.Series])
@parametrize("random_state", [0])
@parametrize("sample_size", [0.25, 0.5, 5, 10])
def test_subsample_validity(X, x_type, y, y_type, random_state, sample_size, task):
    """Asserts the validity of the function with all valid types

    We want to make sure that `subsample` works correctly with all the types listed
    as x_type and y_type.

    We also want to make sure it works with all kinds of target types.

    The output should maintain the types, and subsample the correct amount.
    """
    assert len(X) == len(y)  # Make sure our test data is correct

    if y_type == pd.Series and task in [
        MULTILABEL_CLASSIFICATION,
        MULTIOUTPUT_REGRESSION,
    ]:
        # We can't have a pd.Series with multiple values as it's 1 dimensional
        pytest.skip("Can't have pd.Series as y when task is n-dimensional")

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
            X,
            y,
            random_state=random_state,
            sample_size=sample_size,
            is_classification=task in CLASSIFICATION_TASKS,
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


@parametrize("X", [np.asarray([[0, 0, 1]] * 10)])
@parametrize(
    "dtype", supported_precision_reductions + [np.dtype("float32"), np.dtype("float64")]
)
@parametrize("x_type", [np.ndarray, csr_matrix])
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
        np.dtype("float32"): np.float32,
        np.dtype("float64"): np.float32,
    }
    if hasattr(np, "float96"):
        expected[np.float96] = np.float64

    if hasattr(np, "float128"):
        expected[np.float128] = np.float64

    assert precision == expected[dtype]

    # Check that X's shape was not modified in any way
    assert X.shape == X_reduced.shape

    # Check that the return type is the one as we passed in
    assert type(X) == type(X_reduced)


@parametrize("X", [np.asarray([0, 0, 1]) * 10])
@parametrize("dtype", [np.int32, np.int64, np.complex128])
def test_reduce_precision_with_unsupported_dtypes(X, dtype):
    X = X.astype(dtype)
    with pytest.raises(ValueError) as err:
        reduce_precision(X)

    expected = (
        f"X.dtype = {X.dtype} not equal to any supported "
        f"{supported_precision_reductions}"
    )

    assert err.value.args[0] == expected


@parametrize(
    "X",
    [
        np.ones(
            (100000, 10), dtype=np.float64
        )  # Make it big for reductions to take place
    ],
)
@parametrize("x_type", [csr_matrix, np.ndarray])
@parametrize("dtype", supported_precision_reductions)
@parametrize(
    "y, is_classification",
    [
        (np.ones((100000,)), True),
        (np.ones((100000,)), False),
    ],
)
@parametrize("memory_allocation", [0.1, 1 / 5.2, 1 / 8, 1])
@parametrize("operations", [["precision"], ["subsample"], ["precision", "subsample"]])
def test_reduce_dataset_reduces_size_and_precision(
    X, x_type, dtype, y, is_classification, memory_allocation, operations
):
    assert len(X) == len(y)
    X = X.astype(dtype)
    if x_type == csr_matrix:
        X = x_type(X)

    random_state = 0
    memory_limit = 2  # Force reductions

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        X_out, y_out = reduce_dataset_size_if_too_large(
            X=X,
            y=y,
            random_state=random_state,
            memory_limit=memory_limit,
            operations=operations,
            memory_allocation=memory_allocation,
            is_classification=is_classification,
        )

    def bytes(arr):
        return arr.nbytes if isinstance(arr, np.ndarray) else arr.data.nbytes

    # If we expect some precision reduction unless at float32 already
    if "precision" in operations and dtype != np.float32:
        expected = reduction_mapping[X.dtype]
        assert X_out.dtype == expected
        assert bytes(X_out) < bytes(X)

    # If we expect some subsampling
    if "subsample" in operations:
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
            operations=["precision"],
            memory_limit=1,
            memory_allocation=0.1,
            is_classification=False,
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
            memory_allocation=0.1,
            is_classification=False,
        )

    expected_err = f"Unknown operation `{invalid_op}`"
    assert err.value.args[0] == expected_err


@parametrize("memory_allocation", [-0.5, 0.0, 1.0, 1.5])
def test_reduce_dataset_invalid_memory_allocation_float(memory_allocation: float):
    """
    Parameters
    ----------
    memory_allocation: float
        An invalid memory allocation as a float

    Expects
    -------
    * Should raise a ValueError
    """
    with pytest.raises(
        ValueError, match=r"memory_allocation if float must be in \(0, 1\)"
    ):
        reduce_dataset_size_if_too_large(
            X=np.empty(1),
            y=np.empty(1),
            memory_limit=100,
            is_classification=True,
            memory_allocation=memory_allocation,
        )


@parametrize("memory_allocation", [-1, 0, 100, 101])
def test_reduce_dataset_invalid_memory_allocation_int(memory_allocation: int):
    """
    Parameters
    ----------
    memory_allocation: float
        An invalid memory allocation as a int

    Expects
    -------
    * Should raise a ValueError
    """
    with pytest.raises(
        ValueError, match=r"memory_allocation if int must be in \(0, memory_limit"
    ):
        reduce_dataset_size_if_too_large(
            X=np.empty(1),
            y=np.empty(1),
            is_classification=True,
            memory_limit=100,
            memory_allocation=memory_allocation,
        )


@parametrize("memory_allocation", ["100", {"a": 1}, [100]])
def test_reduce_dataset_invalid_memory_allocation_type(memory_allocation: Any):
    """
    Parameters
    ----------
    memory_allocation: Any
        An invalid memory allocation type

    Expects
    -------
    * Should raise a ValueError
    """
    with pytest.raises(ValueError, match=r"Unknown type for `memory_allocation`"):
        reduce_dataset_size_if_too_large(
            X=np.empty(1),
            y=np.empty(1),
            memory_limit=100,
            is_classification=True,
            memory_allocation=memory_allocation,
        )


@pytest.mark.parametrize(
    "memory_limit,precision,task",
    [
        (memory_limit, precision, task)
        for task in chain(CLASSIFICATION_TASKS, REGRESSION_TASKS)
        for precision in (float, np.float32, np.float64, np.float128)
        for memory_limit in (1, 100)
    ],
)
def test_reduce_dataset_subsampling_explicit_values(memory_limit, precision, task):
    random_state = 0
    fixture = {
        BINARY_CLASSIFICATION: {
            1: {float: 2621, np.float32: 2621, np.float64: 2621, np.float128: 1310},
            100: {
                float: 12000,
                np.float32: 12000,
                np.float64: 12000,
                np.float128: 12000,
            },
        },
        MULTICLASS_CLASSIFICATION: {
            1: {float: 409, np.float32: 409, np.float64: 409, np.float128: 204},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        MULTILABEL_CLASSIFICATION: {
            1: {float: 409, np.float32: 409, np.float64: 409, np.float128: 204},
            100: {float: 1797, np.float32: 1797, np.float64: 1797, np.float128: 1797},
        },
        REGRESSION: {
            1: {float: 1310, np.float32: 1310, np.float64: 1310, np.float128: 655},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        },
        MULTIOUTPUT_REGRESSION: {
            1: {float: 1310, np.float32: 1310, np.float64: 1310, np.float128: 655},
            100: {float: 5000, np.float32: 5000, np.float64: 5000, np.float128: 5000},
        },
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
            X=X,
            y=y,
            random_state=random_state,
            memory_limit=memory_limit,
            is_classification=task in CLASSIFICATION_TASKS,
            operations=["precision", "subsample"],
            memory_allocation=0.1,
        )

    # Assert the new number of samples
    assert X_new.shape[0] == fixture[task][memory_limit][precision]

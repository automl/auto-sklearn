from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.pipeline.util import get_dataset

from pytest import fixture

from test.conftest import DEFAULT_SEED

Data = Tuple[SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES]


def astype(
    t: np.ndarray | list | csr_matrix | pd.DataFrame | pd.Series,
    x: Any,
) -> Any:
    """Convert data to allowed types"""
    if t == np.ndarray:
        return np.asarray(x)
    else:
        return t(x)  # type: ignore


# TODO Remove the implementation in autosklearn.pipeline.util and just put here
@fixture
def make_sklearn_dataset() -> Callable:
    """
    Parameters
    ----------
    name : str = "iris"
        Name of the dataset to get

    make_sparse : bool = False
        Wehther to make the data sparse

    add_NaNs : bool = False
        Whether to add NaNs to the data

    train_size_maximum : int = 150
        THe maximum size of training data

    make_multilabel : bool = False
        Whether to force the data into being multilabel

    make_binary : bool = False
        Whether to force the data into being binary

    task: Optional[int] = None
        The task of the data, required for the datamanager

    feat_type: Optional[Dict | str] = None
        The features types for the data if making a XYDataManager

    as_datamanager: bool = False
        Wether to return the information as an XYDataManager

    Returns
    -------
    (X_train, Y_train, X_test, Y_Test) | XYDataManager
    """

    def _make(
        name: str = "iris",
        make_sparse: bool = False,
        add_NaNs: bool = False,
        train_size_maximum: int = 150,
        make_multilabel: bool = False,
        make_binary: bool = False,
        task: Optional[int] = None,
        feat_type: Optional[Dict | str] = None,
        as_datamanager: bool = False,
        return_target_as_string: bool = False,
    ) -> Any:
        X, y, Xt, yt = get_dataset(
            dataset=name,
            make_sparse=make_sparse,
            add_NaNs=add_NaNs,
            train_size_maximum=train_size_maximum,
            make_multilabel=make_multilabel,
            make_binary=make_binary,
            return_target_as_string=return_target_as_string,
        )

        if not as_datamanager:
            return (X, y, Xt, yt)
        else:

            assert task is not None and feat_type is not None
            if isinstance(feat_type, str):
                feat_type = {i: feat_type for i in range(X.shape[1])}

            return XYDataManager(
                X,
                y,
                Xt,
                yt,
                task=task,
                dataset_name=name,
                feat_type=feat_type,
            )

    return _make


def _make_binary_data(
    dims: Tuple[int, ...] = (100, 3),
    weights: Optional[Sequence[float] | np.ndarray] = None,
    types: Tuple[
        np.ndarray | csr_matrix | pd.DataFrame | list,
        np.ndarray | csr_matrix | pd.DataFrame | list | pd.Series,
    ] = (np.ndarray, np.ndarray),
    random_state: int | np.random.RandomState = DEFAULT_SEED,
) -> Data:
    X_type, y_type = types
    rs = check_random_state(random_state)

    classes = [0, 1]

    if not weights:
        weights = np.ones_like(classes) / len(classes)

    assert len(weights) == len(classes)
    weights = weights / np.sum(weights, keepdims=True)

    X = rs.rand(*dims)
    y = rs.choice([0, 1], dims[0], p=weights)

    return astype(X_type, X), astype(y_type, y)


def _make_multiclass_data(
    dims: Tuple[int, ...] = (100, 3),
    classes: int | np.ndarray | List = 3,
    weights: Optional[np.ndarray | List[float]] = None,
    types: Tuple[
        np.ndarray | csr_matrix | pd.DataFrame | list,
        np.ndarray | csr_matrix | pd.DataFrame | list | pd.Series,
    ] = (np.ndarray, np.ndarray),
    random_state: int | np.random.RandomState = DEFAULT_SEED,
) -> Data:
    X_type, y_type = types

    if isinstance(classes, int):
        classes = np.asarray(list(range(classes)))

    rs = check_random_state(random_state)

    if not weights:
        weights = np.ones_like(classes) / len(classes)

    assert len(weights) == len(classes)
    weights = weights / np.sum(weights, keepdims=True)

    X = rs.rand(*dims)
    y = rs.choice(classes, dims[0], p=weights)

    return astype(X_type, X), astype(y_type, y)


def _make_multilabel_data(
    dims: Tuple[int, ...] = (100, 3),
    classes: np.ndarray | List = [[0, 0], [0, 1], [1, 0], [1, 1]],
    weights: Optional[np.ndarray | List[float]] = None,
    types: Tuple[
        np.ndarray | csr_matrix | pd.DataFrame | list,
        np.ndarray | csr_matrix | pd.DataFrame | list | pd.Series,
    ] = (np.ndarray, np.ndarray),
    random_state: int | np.random.RandomState = DEFAULT_SEED,
) -> Data:
    X_type, y_type = types

    classes = np.asarray(classes)
    assert classes.ndim > 1 and classes.shape[1] > 1

    rs = check_random_state(random_state)

    # Weights indicate each label tuple, and not the weights of individual labels
    # in that tuple
    if not weights:
        weights = np.ones(classes.shape[0]) / len(classes)

    assert len(weights) == len(classes)
    weights = weights / np.sum(weights, keepdims=True)

    X = rs.rand(*dims)

    class_indices = rs.choice(len(classes), dims[0], p=weights)
    y = classes[class_indices]

    return astype(X_type, X), astype(y_type, y)


def _make_regression_data(
    dims: Tuple[int, ...] = (100, 3),
    types: Tuple[
        np.ndarray | csr_matrix | pd.DataFrame | list,
        np.ndarray | csr_matrix | pd.DataFrame | list | pd.Series,
    ] = (np.ndarray, np.ndarray),
    random_state: int | np.random.RandomState = DEFAULT_SEED,
) -> Data:
    X_type, y_type = types
    rs = check_random_state(random_state)

    if X_type == csr_matrix:
        X = rs.choice([0, 1], dims)
    else:
        X = rs.rand(*dims)

    y = rs.rand(dims[0])

    return astype(X_type, X), astype(y_type, y)


def _make_multioutput_regression_data(
    dims: Tuple[int, ...] = (100, 3),
    targets: int = 2,
    types: Tuple[
        np.ndarray | csr_matrix | pd.DataFrame | list,
        np.ndarray | csr_matrix | pd.DataFrame | list | pd.Series,
    ] = (np.ndarray, np.ndarray),
    random_state: int | np.random.RandomState = DEFAULT_SEED,
) -> Data:
    X_type, y_type = types

    rs = check_random_state(random_state)

    if X_type == csr_matrix:
        X = rs.choice([0, 1], dims)
    else:
        X = rs.rand(*dims)

    y = rs.rand(dims[0], targets)

    return astype(X_type, X), astype(y_type, y)


@fixture
def make_data() -> Callable[..., Data]:
    """Generate some arbitrary x,y data

    Parameters
    ----------
    kind: int = BINARY_CLASSIFICATION
        The task type, one of BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, ...

    **kwargs: Any
        See the corresponding `_make_<x>`

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The generated data
    """

    def _make(
        kind: int = BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> Data:
        dispatches = {
            BINARY_CLASSIFICATION: _make_binary_data,
            MULTICLASS_CLASSIFICATION: _make_multiclass_data,
            MULTILABEL_CLASSIFICATION: _make_multilabel_data,
            REGRESSION: _make_regression_data,
            MULTIOUTPUT_REGRESSION: _make_multioutput_regression_data,
        }

        f = dispatches[kind]
        return f(**kwargs)  # type: ignore

    return _make

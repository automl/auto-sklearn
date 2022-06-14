"""Test things related to how AutoML compresses the dataset size"""
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from autosklearn.automl import AutoML
from autosklearn.constants import BINARY_CLASSIFICATION

from pytest_cases import parametrize
from unittest.mock import patch

from test.util import skip


@parametrize("dataset_compression", [{"methods": ["precision", "subsample"]}])
def test_fit_performs_dataset_compression_without_precision_when_int(
    dataset_compression: Dict,
    make_automl: Callable[..., AutoML],
) -> None:
    """
    Parameters
    ----------
    dataset_compression: Dict
        The dataset_compression arg with "precision" set in it

    Fixtures
    --------
    make_automl: Callable[..., AutoML]
        Makes an automl instance


    Expects
    -------
    * Should call reduce_dataset_size_if_too_large
    * "precision" should have been removed from the "methods" passed to the keyword
        argument "operations" of `reduce_dataset_size_if_too_large`.

    Note
    ----
    * Only done with int's as we can't reduce precision of ints in a meaningful way
    """
    X = np.ones((100, 10), dtype=int)
    y = np.random.random((100,))

    auto = make_automl(dataset_compression=dataset_compression)

    with patch(
        "autosklearn.automl.reduce_dataset_size_if_too_large", return_value=(X, y)
    ) as mck:
        # To prevent fitting anything we use `only_return_configuration_space`
        auto.fit(X, y, only_return_configuration_space=True, task=BINARY_CLASSIFICATION)

    assert mck.call_count == 1

    args, kwargs = mck.call_args
    assert kwargs["operations"] == ["subsample"]


@parametrize(
    "X_type",
    [
        np.ndarray,
        csr_matrix,
        skip(
            list,
            "dataset_compression does not support pandas types yet and list gets"
            " converted in InputValidator",
        ),
        skip(pd.DataFrame, "dataset_compression does not support pandas types yet"),
    ],
)
@parametrize(
    "y_type",
    [
        np.ndarray,
        skip(csr_matrix, "See TODO note in `test_fit_performs_dataset_compression`"),
        list,
        skip(pd.DataFrame, "dataset_compression does not support pandas types yet"),
        skip(pd.Series, "dataset_compression does not support pandas types yet"),
    ],
)
def test_fit_performs_dataset_compression(
    X_type: Any,
    y_type: Any,
    make_automl: Callable[..., AutoML],
    make_data: Callable[..., Any],
) -> None:
    """
    Parameters
    ----------
    mock_reduce_dataset: MagicMock
        A mock function to view call

    X_type: Union[np.ndarray, csr_matrix, list, pd.Dataframe]
        Feature to reduce

    y_type: Union[np.ndarray, csr_matrix, list, pd.Series, pd.Dataframe]
        Target to reduce (regression values)

    Fixtures
    --------
    make_automl: Callable[..., AutoML]
        Factory to make automl instance

    make_data: Callable
        Factory to make data

    Expects
    -------
    * Should call reduce_dataset_size_if_too_large

    # TODO not sure how to keep function behaviour and just use the mock object so
    # that we can assert it was called.
    #
    # * `fit` will convert sparse `y`
    # * This gets passed to `reduce_dataset_size_if_too_large`
    # * The de-sparsified `y` is required for the datamanager later on
    #
    # Mocking away the functionality and just returning the X, y we see here will means
    # that the datamanager will get the sparse y and crash, hence we manually convert
    # here
    """
    X, y = make_data(types=(X_type, y_type))

    auto = make_automl(dataset_compression=True)

    with patch(
        "autosklearn.automl.reduce_dataset_size_if_too_large", return_value=(X, y)
    ) as mck:
        # To prevent fitting anything we use `only_return_configuration_space`
        auto.fit(X, y, only_return_configuration_space=True, task=BINARY_CLASSIFICATION)

    assert mck.called
    del auto

"""Test the _model_predict helper function such that it shapes output correctly"""
from typing import Callable, Dict, Tuple

import warnings

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from autosklearn.automl import _model_predict
from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)
from autosklearn.util.logging_ import PicklableClientLogger

from pytest_cases import parametrize


class WarningModel:
    """Simple model that returns incorrect shape and issues warning"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Shout a warning during prediction"""
        warnings.warn("shout")
        return X


@parametrize(
    "dataspec, expected_shape",
    [
        ({"kind": BINARY_CLASSIFICATION, "dims": (100, 5)}, (100, 2)),
        ({"kind": MULTICLASS_CLASSIFICATION, "dims": (100, 5), "classes": 3}, (100, 3)),
        (
            {
                "kind": MULTILABEL_CLASSIFICATION,
                "dims": (100, 5),
                "classes": [[0, 0], [0, 1], [1, 0], [1, 1]],
            },
            (100, 2),  # TODO seems wrong
        ),
    ],
)
def test_classifier_output_shape(
    dataspec: Dict,
    expected_shape: Tuple[int, ...],
    make_voting_classifier: Callable[..., VotingClassifier],
    make_data: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Parameters
    ----------
    dataspec : Dict
        The spec to make data of

    expected_shape : Tuple[int, ...]
        The expected shape of the output of _model_predict

    Fixtures
    --------
    make_voting_classifier : Callable[..., VotingClassifier]
        Factory to make a voting classifier which _model_predict expects

    make_data : Callable[..., Tuple[np.ndarray, np.ndarray]]
        Factory to make data according to a spec

    Expects
    -------
    * The output shape after predicting should be the expected shape

    Note
    ----
    * The output shape for MULTILABEL_CLASSIFICATION seems wrong according to

    """
    task = dataspec["kind"]
    X, y = make_data(**dataspec)

    voter = make_voting_classifier(X=X, y=y)

    output = _model_predict(voter, X, task=task)
    assert output.shape == expected_shape


@parametrize(
    "dataspec, expected_shape",
    [
        ({"kind": REGRESSION, "dims": (100, 5)}, (100,)),
        ({"kind": MULTIOUTPUT_REGRESSION, "dims": (100, 5), "targets": 3}, (100, 3)),
    ],
)
def test_regressor_output_shape(
    dataspec: Dict,
    expected_shape: Tuple[int, ...],
    make_voting_regressor: Callable[..., VotingRegressor],
    make_data: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> None:
    """
    Parameters
    ----------
    dataspec : Dict
        The spec to make data of

    expected_shape : Tuple[int, ...]
        The expected shape of the output of _model_predict

    Fixtures
    --------
    make_voting_regressor: Callable[..., VotingRegressor]
        Factory to make a voting classifier which _model_predict expects

    make_data : Callable[..., Tuple[np.ndarray, np.ndarray]]
        Factory to make data according to a spec
    """
    task = dataspec["kind"]
    X, y = make_data(**dataspec)

    voter = make_voting_regressor(X=X, y=y)

    output = _model_predict(voter, X, task=task)
    assert output.shape == expected_shape


def test_outputs_warnings_to_logs(
    mock_logger: PicklableClientLogger,
) -> None:
    """
    Fixtures
    --------
    mock_logger : PicklableClientLogger
        A mock logger that can be queried for call counts

    Expects
    -------
    * Any warning emitted by a model should be redirected to the logger
    """
    _model_predict(
        model=WarningModel(), X=np.eye(5), task=REGRESSION, logger=mock_logger
    )

    assert mock_logger.warning.call_count == 1  # type: ignore


def test_outputs_to_stdout_if_no_logger() -> None:
    """
    Expects
    -------
    * With no logger, any warning emitted by a model goes to standard out
    """
    with warnings.catch_warnings(record=True) as w:
        _model_predict(model=WarningModel(), X=np.eye(5), task=REGRESSION, logger=None)

    assert len(w) == 1, "One warning sould have been emmited"

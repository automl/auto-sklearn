"""Test the refitting functionality of an automl instance"""
from typing import Callable, Union

from itertools import repeat

import numpy as np

from autosklearn.automl import AutoML
from autosklearn.data.validation import InputValidator

from pytest_cases import parametrize
from unittest.mock import Mock


@parametrize("budget_type", [None, "iterations"])
def test_shuffle_on_fail(
    budget_type: Union[None, str],
    make_automl: Callable[..., AutoML],
) -> None:
    """
    Parameters
    ----------
    budget_type : Union[None, str]
        The budget type to use

    Fixtures
    --------
    make_automl : Callable[..., AutoML]
        Factory to make an AutoML instance

    Expects
    -------
    * The automl should not be able to fit before `refit`
    * The model should be attempted to fitted `n_fails` times before successing once
     after
     * The automl should be able to fit after `refit`
    """
    n_fails = 3
    failing_model = Mock()
    failing_model.fit.side_effect = [ValueError()] * n_fails + [None]  # type: ignore
    failing_model.estimator_supports_iterative_fit.side_effect = repeat(False)

    ensemble_mock = Mock()
    ensemble_mock.get_selected_model_identifiers.return_value = [(1, 1, 50.0)]

    X = np.ones((3, 2))
    y = np.ones((3,))

    input_validator = InputValidator()
    input_validator.fit(X, y)

    auto = make_automl()
    auto.ensemble_ = ensemble_mock  # type: ignore
    auto.models_ = {(1, 1, 50.0): failing_model}
    auto._budget_type = budget_type
    auto.InputValidator = input_validator

    assert not auto._can_predict
    auto.refit(X, y)

    assert failing_model.fit.call_count == n_fails + 1
    assert auto._can_predict

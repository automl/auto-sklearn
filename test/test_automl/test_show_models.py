"""Test the show models functinality of an automl instance"""
from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases

import test.test_automl.cases as cases


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "no_ensemble"])
def test_no_ensemble_produces_empty_show_models(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        The automl object with no ensemble size to test

    Expects
    -------
    * Show models should return an empty dict
    """
    assert automl.show_models() == {}

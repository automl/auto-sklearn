import numpy as np

from autosklearn.automl import AutoML

import pytest
from pytest_cases import filters as ft
from pytest_cases import parametrize, parametrize_with_cases

import test.test_automl.cases as cases


@parametrize("ensemble_size", [-10, -1, 0])
@parametrize_with_cases("automl", cases=cases, filter=~ft.has_tag("fitted"))
def test_non_positive_ensemble_size_raises(
    tmp_dir: str,
    automl: AutoML,
    ensemble_size: int,
) -> None:
    """
    Parameters
    ----------
    automl: AutoML
        The AutoML object to test

    ensemble_size : int
        The ensemble size to use

    Expects
    -------
    * Can't fit ensemble with non-positive ensemble size
    """
    dummy_data = np.array([1, 1, 1])

    with pytest.raises(ValueError):
        automl.fit_ensemble(dummy_data, ensemble_size=ensemble_size)

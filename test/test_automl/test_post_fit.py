from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases

import test.test_automl.cases as cases


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "holdout"])
def test_holdout_loaded_models(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        The fitted automl object to test

    Expects
    -------
    * The ensemble should not be empty
    * The models_ should contain the identifiers of what's in the ensemble
    * The cv_models_ attr should remain None
    """
    assert automl.ensemble_ is not None

    ensemble_identifiers = automl.ensemble_.get_selected_model_identifiers()

    assert set(automl.models_.keys()) == set(ensemble_identifiers)
    assert automl.cv_models_ is None


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "cv"])
def test_cv_loaded_models(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        The fitted automl object to test

    Expects
    -------
    * The ensemble should not be empty
    * The models_ should contain the identifiers of what's in the ensemble
    * The cv_models_ should contain the identifiers of what's in the ensemble
    """
    assert automl.ensemble_ is not None

    ensemble_identifiers = automl.ensemble_.get_selected_model_identifiers()

    assert set(automl.models_.keys()) == set(ensemble_identifiers)
    assert set(automl.cv_models_.keys()) == set(ensemble_identifiers)


@parametrize_with_cases("automl", cases=cases, has_tag=["fitted", "no_ensemble"])
def test_no_ensemble(automl: AutoML) -> None:
    """
    Parameters
    ----------
    automl : AutoML
        A fitted automl object with ensemble size specified as 0

    Expects
    -------
    * Auto-sklearn does not load a model
    * The models_ should be of size 0
    * The cv_models_ should remain None
    """
    assert automl.ensemble_ is None
    assert len(automl.models_) == 0
    assert len(automl.cv_models_) == 0

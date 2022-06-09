"""Test the performance of automl instances after fitting"""

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from autosklearn.automl import AutoML

from pytest_cases import parametrize_with_cases

import test.test_automl.cases as cases


@parametrize_with_cases("automl", cases.case_classifier_fitted_holdout_multiobjective)
def test_performance_with_multiobjective(automl: AutoML) -> None:
    """
    Expects
    -------
    * Auto-sklearn can predict/predict_proba and has a model
    * Each ensemble in the pareto_set can predict/predict_proba
    """
    # TODO: This test is hyperspecific to this one case
    #
    #   Long term we probably want to return additional info about the case so we can
    #   test things for other than this case

    # Check that the predict function works
    X = np.array([[1.0, 1.0, 1.0, 1.0]])

    assert automl.predict_proba(X).shape == (1, 3)
    assert automl.predict(X).shape == (1,)

    pareto_front = automl._load_pareto_set()
    for ensemble in pareto_front:

        assert isinstance(ensemble, (VotingClassifier, VotingRegressor))

        y_pred = ensemble.predict_proba(X)
        assert y_pred.shape == (1, 3)

        y_pred = ensemble.predict(X)
        assert y_pred in ["setosa", "versicolor", "virginica"]

    statistics = automl.sprint_statistics()
    assert "Metrics" in statistics
    assert ("Best validation score: 0.9" in statistics) or (
        "Best validation score: 1.0" in statistics
    ), statistics

import sklearn.ensemble

from autosklearn.pipeline.components.regression.gradient_boosting import \
    GradientBoosting

from .test_base import BaseRegressionComponentTest


class GradientBoostingComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.8377141058954396
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.8377141058954396
    res["default_boston_sparse"] = None
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.33468007022597324
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.33468007022597324
    res["default_diabetes_sparse"] = None
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.ensemble.GradientBoostingRegressor
    module = GradientBoosting
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }
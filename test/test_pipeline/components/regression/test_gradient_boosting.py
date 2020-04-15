import sklearn.ensemble

from autosklearn.pipeline.components.regression.gradient_boosting import \
    GradientBoosting

from .test_base import BaseRegressionComponentTest


class GradientBoostingComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.7677915076432977
    res["default_boston_sparse"] = None
    res["default_diabetes"] = 0.3657740311481247
    res["default_diabetes_sparse"] = None

    sk_mod = sklearn.ensemble.GradientBoostingRegressor
    module = GradientBoosting
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }

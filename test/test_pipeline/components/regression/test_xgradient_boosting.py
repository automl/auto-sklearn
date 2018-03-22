import autosklearn.pipeline.implementations.xgb

from autosklearn.pipeline.components.regression.xgradient_boosting import \
    XGradientBoostingRegressor

from .test_base import BaseRegressionComponentTest


class XGradientBoostingComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8125472788212348
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.812547281171597
    res["default_boston_sparse"] = 0.5372987061524691
    res["default_boston_iterative_sparse"] = 0.53729870534358
    res["default_diabetes"] = 0.3342686707807938
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.3342686736551591
    res["default_diabetes_sparse"] = 0.20263166256483256
    res["default_diabetes_iterative_sparse"] = 0.20263166256483256

    sk_mod = autosklearn.pipeline.implementations.xgb.CustomXGBRegressor
    module = XGradientBoostingRegressor
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 128,
    }
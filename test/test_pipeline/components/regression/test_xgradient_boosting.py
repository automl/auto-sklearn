import autosklearn.pipeline.implementations.xgb

from autosklearn.pipeline.components.regression.xgradient_boosting import \
    XGradientBoostingRegressor

from .test_base import BaseRegressionComponentTest


class XGradientBoostingComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8117876904342379
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = 0.8117876904342379
    res["default_boston_sparse"] = 0.5097717471204916
    res["default_boston_iterative_sparse"] = 0.5097717471204916
    res["default_diabetes"] = 0.31922603199904087
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.31922603199904087
    res["default_diabetes_sparse"] = 0.1983459101561711
    res["default_diabetes_iterative_sparse"] = 0.1983459101561711

    sk_mod = autosklearn.pipeline.implementations.xgb.CustomXGBRegressor
    module = XGradientBoostingRegressor
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 512,
    }
import autosklearn.pipeline.implementations.xgb

from autosklearn.pipeline.components.regression.xgradient_boosting import \
    XGradientBoostingRegressor

from .test_base import BaseRegressionComponentTest


class XGradientBoostingComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8155209334566791
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.8155209334566791
    res["default_boston_sparse"] = 0.5734978224089335
    res["default_boston_iterative_sparse"] = 0.5734978224089335
    res["default_diabetes"] = 0.29100776654206073
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.29100776654206073
    res["default_diabetes_sparse"] = 0.1996773189850003
    res["default_diabetes_iterative_sparse"] = 0.1996773189850003
    res['ignore_hps'] = ['n_estimators']

    sk_mod = autosklearn.pipeline.implementations.xgb.CustomXGBRegressor
    module = XGradientBoostingRegressor

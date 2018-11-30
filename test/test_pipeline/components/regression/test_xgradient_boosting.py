import autosklearn.pipeline.implementations.xgb

from autosklearn.pipeline.components.regression.xgradient_boosting import \
    XGradientBoostingRegressor

from .test_base import BaseRegressionComponentTest


class XGradientBoostingComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.7950690273856177
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.7950690350340658
    res["default_boston_sparse"] = 0.4089636428137894
    res["default_boston_iterative_sparse"] = 0.40896364129803287
    res["default_diabetes"] = 0.3252009519763832
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.3252009519763832
    res["default_diabetes_sparse"] = 0.15356041856907898
    res["default_diabetes_iterative_sparse"] = 0.15356041856907898
    res['ignore_hps'] = ['n_estimators']

    sk_mod = autosklearn.pipeline.implementations.xgb.CustomXGBRegressor
    module = XGradientBoostingRegressor

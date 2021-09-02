import sklearn.ensemble

from autosklearn.pipeline.components.regression.adaboost import \
    AdaboostRegressor
from .test_base import BaseRegressionComponentTest


class AdaBoostComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.5951486466070626
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = 0.18067558132702222
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.250565253614339
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.09126705185668416
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.ensemble.AdaBoostRegressor

    module = AdaboostRegressor

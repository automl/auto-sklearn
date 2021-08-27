import sklearn.ensemble

from autosklearn.pipeline.components.regression.adaboost import \
    AdaboostRegressor
from .test_base import BaseRegressionComponentTest


class AdaBoostComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.6107915398289729
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = 0.21698819209658282
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.260112022454684
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.09267928398529657
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.ensemble.AdaBoostRegressor

    module = AdaboostRegressor

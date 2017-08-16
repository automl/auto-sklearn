import sklearn.ensemble

from autosklearn.pipeline.components.regression.adaboost import \
    AdaboostRegressor
from .test_base import BaseRegressionComponentTest


class AdaBoostComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.60525743737887405
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = 0.22111559712318207
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.25129853514492517
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.090755670764629537
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.ensemble.AdaBoostRegressor

    module = AdaboostRegressor

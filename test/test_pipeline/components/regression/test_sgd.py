import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = -5.4808512936980714e+31
    res["default_boston_iterative"] = -5.4808512936980714e+31
    res["default_boston_sparse"] = -9.432255366952963e+29
    res["default_boston_iterative_sparse"] = -9.432255366952963e+29
    res["default_diabetes"] = 0.066576586105546731
    res["default_diabetes_iterative"] = 0.066576586105546731
    res["default_diabetes_sparse"] = 0.098980579505685062
    res["default_diabetes_iterative_sparse"] = 0.098980579505685062

    sk_mod = sklearn.linear_model.SGDRegressor

    module = SGD
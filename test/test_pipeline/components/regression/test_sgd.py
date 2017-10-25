import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = -7.76508935746e+30
    res["default_boston_iterative"] = -7.76508935746e+30
    res["default_boston_sparse"] = -8.4314088748670574e+30
    res["default_boston_iterative_sparse"] = -8.4314088748670574e+30
    res["default_diabetes"] = 0.28217342878579466
    res["default_diabetes_iterative"] = 0.28217342878579466
    res["default_diabetes_sparse"] = 0.099141102939435899
    res["default_diabetes_iterative_sparse"] = 0.099141102939435899

    sk_mod = sklearn.linear_model.SGDRegressor

    module = SGD
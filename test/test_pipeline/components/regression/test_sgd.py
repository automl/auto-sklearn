import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = -4.19960551863e+26
    res["default_boston_iterative"] = -4.19960551863e+26
    res["default_boston_sparse"] = -3.4078908871512885e+27
    res["default_boston_iterative_sparse"] = -3.4078908871512885e+27
    res["default_diabetes"] = 0.31775104100149354
    res["default_diabetes_iterative"] = 0.31775104100149354
    res["default_diabetes_sparse"] = 0.10849537848918589
    res["default_diabetes_iterative_sparse"] = 0.10849537848918589

    sk_mod = sklearn.linear_model.SGDRegressor

    module = SGD
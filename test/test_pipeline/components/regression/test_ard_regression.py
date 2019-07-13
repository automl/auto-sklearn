import sklearn.linear_model

from autosklearn.pipeline.components.regression.ard_regression import \
    ARDRegression
from .test_base import BaseRegressionComponentTest


class ARDRegressionComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.7033160711079323
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = None
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.4172008418124077
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = None
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.linear_model.ARDRegression

    module = ARDRegression

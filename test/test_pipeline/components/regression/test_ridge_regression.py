import sklearn.linear_model

from autosklearn.pipeline.components.regression.ridge_regression import \
    RidgeRegression
from .test_base import BaseRegressionComponentTest


class RidgeComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.70337988453496891
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = 0.11243478302989141
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.32614416980439365
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.12989713186102791
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.linear_model.Ridge

    module = RidgeRegression

import sklearn.linear_model

from autosklearn.pipeline.components.regression.libsvm_svr import LibSVM_SVR
from .test_base import BaseRegressionComponentTest


class SupportVectorComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = -0.030006883949312613
    res["default_boston_iterative"] = None
    res["default_boston_sparse"] = -0.062749211736050192
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.12849591861430087
    res["default_diabetes_iterative"] = None
    res["default_diabetes_sparse"] = 0.0098877566961463881
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.svm.SVR

    module = LibSVM_SVR
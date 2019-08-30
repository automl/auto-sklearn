import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    # Values are extremly bad because the invscaling does not drop the
    # learning rate aggressively enough!
    res = dict()
    res["default_boston"] = -1.1406255966671184e+28
    res["boston_n_calls"] = 6
    res["default_boston_iterative"] = -5.811679712715771e+27
    res["default_boston_sparse"] = -2.5492867904339725e+28
    res["default_boston_iterative_sparse"] = -4.3762864606281644e+27
    res["default_diabetes"] = 0.24813492281662985
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.24813331594531207
    res["default_diabetes_sparse"] = 0.03484084539994714
    res["default_diabetes_iterative_sparse"] = 0.0725405368884059

    sk_mod = sklearn.linear_model.SGDRegressor

    module = SGD
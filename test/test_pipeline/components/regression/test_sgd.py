import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    # Values are extremly bad because the invscaling does not drop the
    # learning rate aggressively enough!
    res = dict()
    res["default_boston"] = -3.3027457155755987e+28
    res["boston_n_calls"] = 4
    res["default_boston_iterative"] = -5.80882066268658e+27
    res["default_boston_sparse"] = -6.022031094018292e+27
    res["default_boston_iterative_sparse"] = -4.3762864606281644e+27
    res["default_diabetes"] = 0.24813492281662985
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.24813331594531207
    res["default_diabetes_sparse"] = 0.03427063871071423
    res["default_diabetes_iterative_sparse"] = 0.0725405368884059

    sk_mod = sklearn.linear_model.SGDRegressor

    module = SGD
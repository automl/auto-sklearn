import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    # Values are extremely bad because the invscaling does not drop the
    # learning rate aggressively enough!
    res = dict()
    res["default_boston"] = -1.665805157779146e+28
    res["boston_n_calls"] = 6
    res["default_boston_iterative"] = res["default_boston"]
    res["default_boston_sparse"] = -1.463839564527922e+28
    res["default_boston_iterative_sparse"] = res["default_boston_sparse"]
    res["default_diabetes"] = 0.2744252571536926
    res["diabetes_n_calls"] = 10
    res["default_diabetes_iterative"] = res["default_diabetes"]
    res["default_diabetes_sparse"] = 0.03485642727564764
    res["default_diabetes_iterative_sparse"] = res["default_diabetes_sparse"]

    sk_mod = sklearn.linear_model.SGDRegressor
    module = SGD

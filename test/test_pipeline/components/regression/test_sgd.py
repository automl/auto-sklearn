import sklearn.linear_model

from autosklearn.pipeline.components.regression.sgd import SGD
from .test_base import BaseRegressionComponentTest


class SGDComponentTest(BaseRegressionComponentTest):
    __test__ = True

    # Values are extremely bad because the invscaling does not drop the
    # learning rate aggressively enough!
    res = dict()
    res["default_boston"] = -1.1406255966671184e+28
    res["boston_n_calls"] = 6
    res["default_boston_iterative"] = -1.1406255966671184e+28
    res["default_boston_sparse"] = -2.5492867904339725e+28
    res["default_boston_iterative_sparse"] = -2.5492867904339725e+28
    res["default_diabetes"] = 0.2731178369559112
    res["diabetes_n_calls"] = 10
    res["default_diabetes_iterative"] = 0.2731178369559112
    res["default_diabetes_sparse"] = 0.03484084539994714
    res["default_diabetes_iterative_sparse"] = 0.03484084539994714

    sk_mod = sklearn.linear_model.SGDRegressor
    module = SGD

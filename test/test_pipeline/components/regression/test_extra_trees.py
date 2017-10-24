import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.77812069925853511
    res["default_boston_iterative"] = 0.77812069925853511
    res["default_boston_sparse"] = 0.33175043290947837
    res["default_boston_iterative_sparse"] = 0.33175043290947837
    res["default_diabetes"] = 0.43359660671062761
    res["default_diabetes_iterative"] = 0.43359660671062761
    res["default_diabetes_sparse"] = 0.28040986328860906
    res["default_diabetes_iterative_sparse"] = 0.28040986328860906

    sk_mod = sklearn.ensemble.ExtraTreesRegressor

    module = ExtraTreesRegressor
import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.84600084385083418
    res["default_boston_iterative"] = 0.84600084385083418
    res["default_boston_sparse"] = 0.4241336980285072
    res["default_boston_iterative_sparse"] = 0.4241336980285072
    res["default_diabetes"] = 0.38869684680884398
    res["default_diabetes_iterative"] = 0.38869684680884398
    res["default_diabetes_sparse"] = 0.25345808769763623
    res["default_diabetes_iterative_sparse"] = 0.25345808769763623

    sk_mod = sklearn.ensemble.ExtraTreesRegressor

    module = ExtraTreesRegressor

import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.85057762155064132
    res["default_boston_iterative"] = 0.84600084385083418
    res["default_boston_sparse"] = 0.40009651469914886
    res["default_boston_iterative_sparse"] = 0.4241336980285072
    res["default_diabetes"] = 0.38997024548179593
    res["default_diabetes_iterative"] = 0.38869684680884398
    res["default_diabetes_sparse"] = 0.25171463472003841
    res["default_diabetes_iterative_sparse"] = 0.25345808769763623

    sk_mod = sklearn.ensemble.ExtraTreesRegressor

    module = ExtraTreesRegressor

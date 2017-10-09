import sklearn.ensemble

from autosklearn.pipeline.components.regression.gradient_boosting import \
    GradientBoosting

from .test_base import BaseRegressionComponentTest


class GradientBoostingComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.83961954550470863
    res["default_boston_iterative"] = 0.83961954550470863
    res["default_boston_sparse"] = None
    res["default_boston_iterative_sparse"] = None
    res["default_diabetes"] = 0.37192663934006487
    res["default_diabetes_iterative"] = 0.37192663934006487
    res["default_diabetes_sparse"] = None
    res["default_diabetes_iterative_sparse"] = None

    sk_mod = sklearn.ensemble.GradientBoostingRegressor

    module = GradientBoosting
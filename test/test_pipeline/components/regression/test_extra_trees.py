import sklearn.ensemble

from autosklearn.pipeline.components.regression.extra_trees import \
    ExtraTreesRegressor
from .test_base import BaseRegressionComponentTest


class ExtraTreesComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.77744792837581866
    res["default_boston_iterative"] = 0.77744792837581866
    res["default_boston_sparse"] = 0.32936702992375644
    res["default_boston_iterative_sparse"] = 0.32936702992375644
    res["default_diabetes"] = 0.43258995365114405
    res["default_diabetes_iterative"] = 0.43258995365114405
    res["default_diabetes_sparse"] = 0.28016012771570553
    res["default_diabetes_iterative_sparse"] = 0.28016012771570553

    sk_mod = sklearn.ensemble.ExtraTreesRegressor

    module = ExtraTreesRegressor
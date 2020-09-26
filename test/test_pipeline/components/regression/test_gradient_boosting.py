import sklearn.ensemble

from autosklearn.pipeline.components.regression.gradient_boosting import \
    GradientBoosting

from .test_base import BaseRegressionComponentTest


class GradientBoostingComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.7491382574462079
    res["default_boston_iterative"] = 0.7491382574462079
    res["default_boston_sparse"] = None
    res["boston_n_calls"] = 9
    res["default_diabetes"] = 0.2872735632261877
    res["default_diabetes_iterative"] = 0.2872735632261877
    res["default_diabetes_sparse"] = None
    res["diabetes_n_call"] = 11

    sk_mod = sklearn.ensemble.GradientBoostingRegressor
    module = GradientBoosting

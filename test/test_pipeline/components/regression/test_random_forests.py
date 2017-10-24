import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.78511903295069552
    res["default_boston_iterative"] = 0.78511903295069552
    res["default_boston_sparse"] = 0.42511238171276622
    res["default_boston_iterative_sparse"] = 0.42511238171276622
    res["default_diabetes"] = 0.41795829411621988
    res["default_diabetes_iterative"] = 0.41795829411621988
    res["default_diabetes_sparse"] = 0.24346318857157412
    res["default_diabetes_iterative_sparse"] = 0.24346318857157412

    sk_mod = sklearn.ensemble.RandomForestRegressor

    module = RandomForest
import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.78435242169129116
    res["default_boston_iterative"] = 0.78435242169129116
    res["default_boston_sparse"] = 0.42374643794982714
    res["default_boston_iterative_sparse"] = 0.42374643794982714
    res["default_diabetes"] = 0.41795829411621988
    res["default_diabetes_iterative"] = 0.41795829411621988
    res["default_diabetes_sparse"] = 0.24225685933770469
    res["default_diabetes_iterative_sparse"] = 0.24225685933770469

    sk_mod = sklearn.ensemble.RandomForestRegressor

    module = RandomForest
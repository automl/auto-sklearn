import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.84268339869543274
    res["default_boston_iterative"] = 0.83555736176073558
    res["default_boston_sparse"] = 0.43023364434667044
    res["default_boston_iterative_sparse"] = 0.40733598090972378
    res["default_diabetes"] = 0.34759241745667557
    res["default_diabetes_iterative"] = 0.34470637712379781
    res["default_diabetes_sparse"] = 0.24593112889855551
    res["default_diabetes_iterative_sparse"] = 0.22910562828064962

    sk_mod = sklearn.ensemble.RandomForestRegressor

    module = RandomForest

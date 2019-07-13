import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8430744432644872
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.8357818969644876
    res["default_boston_sparse"] = 0.4384357259247913
    res["default_boston_iterative_sparse"] = 0.40733598090972378
    res["default_diabetes"] = 0.34006418275532946
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.3435242905853275
    res["default_diabetes_sparse"] = 0.24065116769491235
    res["default_diabetes_iterative_sparse"] = 0.22973288076916099

    sk_mod = sklearn.ensemble.RandomForestRegressor
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }

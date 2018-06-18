import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8428696878298745
    res["boston_n_calls"] = 7
    res["default_boston_iterative"] = 0.83555736176073558
    res["default_boston_sparse"] = 0.43842300527656664
    res["default_boston_iterative_sparse"] = 0.40733598090972378
    res["default_diabetes"] = 0.3405963933858086
    res["diabetes_n_calls"] = 7
    res["default_diabetes_iterative"] = 0.34470637712379781
    res["default_diabetes_sparse"] = 0.24119350585906507
    res["default_diabetes_iterative_sparse"] = 0.22910562828064962

    sk_mod = sklearn.ensemble.RandomForestRegressor
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }

import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import \
    RandomForest
from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.841295107582696
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = 0.8418168483834201
    res["default_boston_sparse"] = 0.4256329804087612
    res["default_boston_iterative_sparse"] = 0.4279815441481293
    res["default_diabetes"] = 0.3438707361168547
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.3410928574942581
    res["default_diabetes_sparse"] = 0.24330715569007455
    res["default_diabetes_iterative_sparse"] = 0.24158474583742073

    sk_mod = sklearn.ensemble.RandomForestRegressor
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': module.get_max_iter(),
    }

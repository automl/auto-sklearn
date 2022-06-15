import sklearn.ensemble

from autosklearn.pipeline.components.regression.random_forest import RandomForest

from .test_base import BaseRegressionComponentTest


class RandomForestComponentTest(BaseRegressionComponentTest):
    __test__ = True

    res = dict()
    res["default_boston"] = 0.8410063895401654
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = res["default_boston"]
    res["default_boston_sparse"] = 0.4194462097407078
    res["default_boston_iterative_sparse"] = res["default_boston_sparse"]
    res["default_diabetes"] = 0.35034589202040234
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = res["default_diabetes"]
    res["default_diabetes_sparse"] = 0.23731783691227226
    res["default_diabetes_iterative_sparse"] = res["default_diabetes_sparse"]

    sk_mod = sklearn.ensemble.RandomForestRegressor
    module = RandomForest
    step_hyperparameter = {
        "name": "n_estimators",
        "value": module.get_max_iter(),
    }

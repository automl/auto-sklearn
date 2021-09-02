import sklearn.neural_network

from autosklearn.pipeline.components.regression.mlp import MLPRegressor

from .test_base import BaseRegressionComponentTest


class MLPComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.2750079862455884
    res["default_boston_places"] = 4
    res["boston_n_calls"] = 8
    res["boston_iterative_n_iter"] = 236
    res["default_boston_iterative"] = res["default_boston"]
    res["default_boston_iterative_places"] = 1
    res["default_boston_sparse"] = -0.10972947168054104
    res["default_boston_sparse_places"] = 6
    res["default_boston_iterative_sparse"] = res["default_boston_sparse"]
    res["default_boston_iterative_sparse_places"] = 6
    res["default_diabetes"] = 0.35917389841850555
    res["diabetes_n_calls"] = 9
    res["diabetes_iterative_n_iter"] = 435
    res["default_diabetes_iterative"] = res["default_diabetes"]
    res["default_diabetes_sparse"] = 0.25573903970369427
    res["default_diabetes_iterative_sparse"] = res["default_diabetes_sparse"]

    sk_mod = sklearn.neural_network.MLPRegressor
    module = MLPRegressor
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

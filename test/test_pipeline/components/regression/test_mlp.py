import sklearn.neural_network

from autosklearn.pipeline.components.regression.mlp import MLPRegressor

from .test_base import BaseRegressionComponentTest


class MLPComponentTest(BaseRegressionComponentTest):

    __test__ = True

    res = dict()
    res["default_boston"] = 0.5572566941424424
    res["boston_n_calls"] = 9
    res["default_boston_iterative"] = 0.3991011585607287
    res["default_boston_iterative_places"] = 2
    res["default_boston_sparse"] = -0.5023509031147615
    res["default_boston_iterative_sparse"] = -0.49785984132922323
    res["default_diabetes"] = 0.39632559079949414
    res["diabetes_n_calls"] = 9
    res["default_diabetes_iterative"] = 0.4402964422631598
    res["default_diabetes_sparse"] = 0.2283137551930371
    res["default_diabetes_iterative_sparse"] = 0.27430032244279856

    sk_mod = sklearn.neural_network.MLPRegressor
    module = MLPRegressor
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

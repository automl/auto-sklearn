import sklearn.neural_network

from autosklearn.pipeline.components.regression.mlp import MLPRegressor

from .test_base import BaseRegressionComponentTest


class MLPComponentTest(BaseRegressionComponentTest):
    # NOTE: `default_boston`
    #
    # Github runners seem to indeterministicly fail `test_boston`
    # meaning 'default_irish_proba_places' needs to be set.
    # There are known differences to occur on different platforms.
    # https://github.com/scikit-learn/scikit-learn/issues/13108#issuecomment-461696681
    #
    # We are assuming results are deterministic on a given platform as locally
    # there is no randomness i.e. performing the same test 100 times yeilds the
    # same predictions 100 times.
    #
    # Github runners indicate that they run on microsoft Azure with DS2-v2.
    # https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#cloud-hosts-for-github-hosted-runners
    #
    # These seem to have consistent CPU's so I'm unsure what the underlying reason
    # for this to randomly fail only sometimes on Github runners
    __test__ = True

    __test__ = True

    res = dict()
    res["default_boston"] = 0.2750079862455884
    res["default_boston_places"] = 3
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

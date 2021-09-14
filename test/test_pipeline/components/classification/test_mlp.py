import sklearn.neural_network

from autosklearn.pipeline.components.classification.mlp import MLPClassifier

from .test_base import BaseClassificationComponentTest


class MLPComponentTest(BaseClassificationComponentTest):
    # NOTE: `default_iris_proba_places`
    #
    # Github runners seem to indeterministicly fail `test_default_iris_proba`
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

    res = dict()
    res["default_iris"] = 0.98
    res["iris_n_calls"] = 7
    res["iris_iterative_n_iter"] = 94
    res["default_iris_iterative"] = res["default_iris"]
    res["default_iris_proba"] = 0.647786774635315
    res["default_iris_proba_places"] = 6
    res["default_iris_sparse"] = 0.42
    res["default_digits"] = 0.8099574984820886
    res["digits_n_calls"] = 7
    res["digits_iterative_n_iter"] = 124
    res["default_digits_iterative"] = res["default_digits"]
    res["default_digits_binary"] = 0.99210686095932
    res["default_digits_multilabel"] = 0.8083000396415946
    res["default_digits_multilabel_proba"] = 0.8096624850657109

    sk_mod = sklearn.neural_network.MLPClassifier
    module = MLPClassifier
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

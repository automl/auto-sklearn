import sklearn.neural_network

from autosklearn.pipeline.components.classification.mlp import MLPClassifier

from .test_base import BaseClassificationComponentTest


class MLPComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.98
    res["iris_n_calls"] = 7
    res["iris_iterative_n_iter"] = 94
    res["default_iris_iterative"] = res["default_iris"]
    res["default_iris_proba"] = 0.647786774635315
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

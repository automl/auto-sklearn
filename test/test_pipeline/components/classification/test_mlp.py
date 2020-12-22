import sklearn.neural_network

from autosklearn.pipeline.components.classification.mlp import MLPClassifier

from .test_base import BaseClassificationComponentTest


class MLPComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.62
    res["iris_n_calls"] = 7
    res["iris_iterative_n_iter"] = 98
    res["default_iris_iterative"] = res["default_iris"]
    res["default_iris_proba"] = 1.360589495897293
    res["default_iris_sparse"] = 0.22
    res["default_digits"] = 0.6095931997571342
    res["digits_n_calls"] = 7
    res["digits_iterative_n_iter"] = 107
    res["default_digits_iterative"] = res["default_digits"]
    res["default_digits_binary"] = 0.9896782027929569
    res["default_digits_multilabel"] = 0.8318638108383578
    res["default_digits_multilabel_proba"] = 0.6948680429155812

    sk_mod = sklearn.neural_network.MLPClassifier
    module = MLPClassifier
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

import sklearn.neural_network

from autosklearn.pipeline.components.classification.mlp import MLPClassifier

from .test_base import BaseClassificationComponentTest


class MLPComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.84
    res["iris_n_calls"] = 9
    res["default_iris_iterative"] = 0.84
    res["default_iris_proba"] = 0.5615969597449075
    res["default_iris_sparse"] = 0.42
    res["default_digits"] = 0.461445051608986
    res["default_digits_places"] = 2
    res["digits_n_calls"] = 9
    res["default_digits_iterative"] = 0.461445051608986
    res["default_digits_iterative_places"] = 2
    res["default_digits_binary"] = 0.978749241044323
    res["default_digits_multilabel"] = 0.2637626340507736
    res["default_digits_multilabel_places"] = 2
    res["default_digits_multilabel_proba"] = 0.8792680626296924

    sk_mod = sklearn.neural_network.MLPClassifier
    module = MLPClassifier
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

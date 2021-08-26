import sklearn.neural_network

from autosklearn.pipeline.components.classification.mlp import MLPClassifier

from .test_base import BaseClassificationComponentTest


class MLPComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 1.0
    res["iris_n_calls"] = 6
    res["iris_iterative_n_iter"] = 60
    res["default_iris_iterative"] = res["default_iris"]
    res["default_iris_proba"] = 0.6433156484365463
    res["default_iris_sparse"] = 0.48
    res["default_digits"] = 0.7650273224043715
    res["digits_n_calls"] = 7
    res["digits_iterative_n_iter"] = 116
    res["default_digits_iterative"] = res["default_digits"]
    res["default_digits_binary"] = 0.9902853673345476
    res["default_digits_multilabel"] = 0.8528712703226736
    res["default_digits_multilabel_proba"] = 0.9826388888888888

    sk_mod = sklearn.neural_network.MLPClassifier
    module = MLPClassifier
    step_hyperparameter = {
        'name': 'n_iter_',
        'value': module.get_max_iter(),
    }

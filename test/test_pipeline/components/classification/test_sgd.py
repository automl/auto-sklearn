import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD
from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.69999999999999996
    res["iris_n_calls"] = 9
    res["default_iris_iterative"] = 0.7
    res["default_iris_proba"] = 0.599538539014923
    res["default_iris_sparse"] = 0.54
    res["default_digits"] = 0.9234972677595629
    res["digits_n_calls"] = 7
    res["default_digits_iterative"] = 0.9234972677595629
    res["default_digits_binary"] = 0.9933211900425015
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1
    res['ignore_hps'] = ['max_iter']

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD

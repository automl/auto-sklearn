import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD
from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.69999999999999996
    res["default_iris_iterative"] = 0.64000000000000001
    res["default_iris_proba"] = 0.6034556233553382
    res["default_iris_sparse"] = 0.54
    res["default_digits"] = 0.91196114146933815
    res["default_digits_iterative"] = 0.91256830601092898
    res["default_digits_binary"] = 0.99514268366727388
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1
    res['ignore_hps'] = ['max_iter']

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD
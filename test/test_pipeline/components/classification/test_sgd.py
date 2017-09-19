import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD

from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.64
    res["default_iris_iterative"] = 0.76
    res["default_iris_proba"] = 12.433959502167845
    res["default_iris_sparse"] = 0.26
    res["default_digits"] = 0.89981785063752273
    res["default_digits_iterative"] = 0.89981785063752273
    res["default_digits_binary"] = 0.9927140255009107
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD
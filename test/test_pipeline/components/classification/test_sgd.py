import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD
from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.97999999999999998
    res["default_iris_iterative"] = 0.9
    res["default_iris_proba"] = 0.29960543868641343
    res["default_iris_sparse"] = 0.54
    res["default_digits"] = 0.92167577413479052
    res["default_digits_iterative"] = 0.92167577413479052
    res["default_digits_binary"] = 0.99332119004250152
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD
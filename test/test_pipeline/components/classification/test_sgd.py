import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD
from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.84
    res["default_iris_iterative"] = 0.9
    res["default_iris_proba"] = 0.5028042799123823
    res["default_iris_sparse"] = 0.54
    res["default_digits"] = 0.92349726775956287
    res["default_digits_iterative"] = 0.92167577413479052
    res["default_digits_binary"] = 0.99392835458409234
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD
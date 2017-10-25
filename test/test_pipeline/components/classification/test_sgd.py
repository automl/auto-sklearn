import sklearn.linear_model

from autosklearn.pipeline.components.classification.sgd import SGD

from .test_base import BaseClassificationComponentTest


class SGDComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.92
    res["default_iris_iterative"] = 0.68
    res["default_iris_proba"] = 0.32146556612186317
    res["default_iris_sparse"] = 0.41999999999999998
    res["default_digits"] = 0.88767455980570731
    res["default_digits_iterative"] = 0.91317547055251969
    res["default_digits_binary"] = 0.9884638737097754
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.linear_model.SGDClassifier
    module = SGD
import sklearn.ensemble

from autosklearn.pipeline.components.classification.adaboost import \
    AdaboostClassifier
from .test_base import BaseClassificationComponentTest


class AdaBoostComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.93999999999999995
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.22452300738472031
    res["default_iris_sparse"] = 0.85999999999999999
    res["default_digits"] = 0.6879174256223437
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.98299939283545845
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.ensemble.AdaBoostClassifier

    module = AdaboostClassifier
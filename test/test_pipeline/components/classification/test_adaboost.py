import sklearn.ensemble

from autosklearn.pipeline.components.classification.adaboost import \
    AdaboostClassifier
from .test_base import BaseClassificationComponentTest


class AdaBoostComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_on_iris"] = 0.93999999999999995
    res["default_on_iris_proba"] = 0.22452300738472031
    res["default_on_iris_sparse"] = 0.85999999999999999
    res["default_binary"] = 0.93564055859137829

    sk_mod = sklearn.ensemble.AdaBoostClassifier

    module = AdaboostClassifier